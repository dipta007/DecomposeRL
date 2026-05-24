import asyncio
import logging
import os
import random
import sys
from dataclasses import dataclass
from typing import List, Optional

from anthropic import AsyncAnthropic
from anthropic import (
    APIConnectionError as AnthropicAPIConnectionError,
    APIError as AnthropicAPIError,
    APITimeoutError as AnthropicAPITimeoutError,
    AuthenticationError as AnthropicAuthError,
    BadRequestError as AnthropicBadRequest,
    InternalServerError as AnthropicInternalServerError,
    NotFoundError as AnthropicNotFoundError,
    PermissionDeniedError as AnthropicPermissionDenied,
    RateLimitError as AnthropicRateLimitError,
)
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
)
from tqdm.asyncio import tqdm_asyncio

_log = logging.getLogger("decomposer.baselines.api")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(
        logging.Formatter("[%(asctime)s api.retry] %(message)s", datefmt="%H:%M:%S")
    )
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False

_RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APITimeoutError,
    APIConnectionError,
    InternalServerError,
    AnthropicRateLimitError,
    AnthropicAPITimeoutError,
    AnthropicAPIConnectionError,
    AnthropicInternalServerError,
)
_NON_RETRYABLE_EXCEPTIONS = (
    BadRequestError,
    AuthenticationError,
    NotFoundError,
    PermissionDeniedError,
    AnthropicBadRequest,
    AnthropicAuthError,
    AnthropicNotFoundError,
    AnthropicPermissionDenied,
)

_RETRY_INITIAL_BACKOFF_S = 1.0
_RETRY_MAX_BACKOFF_S = 60.0

DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5",
}

VALID_PROVIDERS = ("openai", "anthropic")


@dataclass
class ApiConfig:
    provider: str
    model: str
    base_url: Optional[str]
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 16768
    max_concurrency: int = 64
    seed: int = 42


def build_config(
    provider: str,
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key_env: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 16768,
    max_concurrency: int = 64,
) -> ApiConfig:
    if provider not in VALID_PROVIDERS:
        raise ValueError(f"Unknown provider: {provider}. Valid: {VALID_PROVIDERS}")

    model = model or DEFAULT_MODELS[provider]

    if provider == "openai":
        api_key_env = api_key_env or "OPENAI_API_KEY"
    else:
        api_key_env = api_key_env or "ANTHROPIC_API_KEY"

    api_key = os.environ.get(api_key_env, "")
    if not api_key:
        raise ValueError(f"Environment variable {api_key_env} is empty or unset")

    return ApiConfig(
        provider=provider,
        model=model,
        base_url=base_url,
        api_key=api_key,
        temperature=temperature,
        max_tokens=max_tokens,
        max_concurrency=max_concurrency,
    )


async def _one_call(
    client: AsyncOpenAI,
    cfg: ApiConfig,
    prompt: str,
    sem: asyncio.Semaphore,
    prompt_idx: int = -1,
) -> str:
    async with sem:
        attempt = 0
        backoff = _RETRY_INITIAL_BACKOFF_S
        while True:
            attempt += 1
            try:
                kwargs = dict(
                    model=cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                if cfg.provider == "openai":
                    kwargs["seed"] = cfg.seed
                resp = await client.chat.completions.create(**kwargs)
                if attempt > 1:
                    _log.info("prompt %d: succeeded on attempt %d", prompt_idx, attempt)
                return resp.choices[0].message.content or ""
            except _NON_RETRYABLE_EXCEPTIONS as e:
                _log.error(
                    "prompt %d: NON-RETRYABLE %s on attempt %d: %s",
                    prompt_idx, type(e).__name__, attempt, str(e)[:200],
                )
                return f"[API_ERROR] {type(e).__name__}: {e}"
            except (_RETRYABLE_EXCEPTIONS + (APIError, Exception)) as e:
                jitter = random.uniform(0, backoff * 0.1)
                sleep_s = min(backoff + jitter, _RETRY_MAX_BACKOFF_S)
                _log.warning(
                    "prompt %d attempt %d %s: %s — sleeping %.1fs",
                    prompt_idx, attempt, type(e).__name__, str(e)[:200], sleep_s,
                )
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, _RETRY_MAX_BACKOFF_S)


async def _one_call_anthropic(
    client: AsyncAnthropic,
    cfg: ApiConfig,
    prompt: str,
    sem: asyncio.Semaphore,
    prompt_idx: int = -1,
) -> str:
    async with sem:
        attempt = 0
        backoff = _RETRY_INITIAL_BACKOFF_S
        while True:
            attempt += 1
            try:
                resp = await client.messages.create(
                    model=cfg.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                parts = []
                for block in resp.content or []:
                    t = getattr(block, "text", None)
                    if t:
                        parts.append(t)
                if attempt > 1:
                    _log.info("prompt %d: succeeded on attempt %d", prompt_idx, attempt)
                return "".join(parts)
            except _NON_RETRYABLE_EXCEPTIONS as e:
                _log.error(
                    "prompt %d: NON-RETRYABLE %s on attempt %d: %s",
                    prompt_idx, type(e).__name__, attempt, str(e)[:200],
                )
                return f"[API_ERROR] {type(e).__name__}: {e}"
            except (_RETRYABLE_EXCEPTIONS + (AnthropicAPIError, Exception)) as e:
                jitter = random.uniform(0, backoff * 0.1)
                sleep_s = min(backoff + jitter, _RETRY_MAX_BACKOFF_S)
                _log.warning(
                    "prompt %d attempt %d %s: %s — sleeping %.1fs",
                    prompt_idx, attempt, type(e).__name__, str(e)[:200], sleep_s,
                )
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, _RETRY_MAX_BACKOFF_S)


async def _run_all(cfg: ApiConfig, prompts: List[str]) -> List[str]:
    sem = asyncio.Semaphore(cfg.max_concurrency)
    if cfg.provider == "anthropic":
        client = AsyncAnthropic(api_key=cfg.api_key, base_url=cfg.base_url)
        tasks = [
            _one_call_anthropic(client, cfg, p, sem, i) for i, p in enumerate(prompts)
        ]
    else:
        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        tasks = [_one_call(client, cfg, p, sem, i) for i, p in enumerate(prompts)]
    return await tqdm_asyncio.gather(*tasks, desc=f"API[{cfg.model}]")


def run_api_inference(cfg: ApiConfig, prompts: List[str]) -> List[str]:
    return asyncio.run(_run_all(cfg, prompts))
