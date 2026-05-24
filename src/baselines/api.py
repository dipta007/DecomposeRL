"""
Thin async client for prompted baselines.

Supports two SDK kinds, dispatched by `provider`:

OpenAI-SDK path (chat completions, OpenAI-shaped response):
- provider="openai"     -> uses OPENAI_API_KEY, default base_url
- provider="anthropic"  -> uses ANTHROPIC_API_KEY, base_url=https://api.anthropic.com/v1/
                           (Anthropic's OpenAI-compatible shim — Messages translated to chat)
- provider="custom"     -> uses --api_base_url and --api_key_env you pass in

Anthropic-SDK path (Messages API, Anthropic-shaped response):
- provider="anthropic_native" -> uses Anthropic Messages API against a configurable
                                 base_url + api_key_env. Required when the upstream proxy
                                 returns Anthropic-shaped bodies (e.g. UMBC's LiteLLM
                                 gateway for Claude models, where /v1/chat/completions
                                 returns a response with .content[0].text instead of
                                 .choices[0].message.content).
- provider="umbc"             -> shortcut: anthropic-native against the UMBC gateway with
                                 sensible defaults (base_url=https://gateway.aws.genai.umbc.edu/,
                                 api_key_env=UMBC_GATEWAY_KEY, model="Claude Haiku 4.5").
"""

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


# Configure a module-level logger that prints to stderr so retries are visible
# even when the parent runner's stdout is captured.
_log = logging.getLogger("decomposer.baselines.api")
if not _log.handlers:
    _h = logging.StreamHandler(sys.stderr)
    _h.setFormatter(
        logging.Formatter("[%(asctime)s api.retry] %(message)s", datefmt="%H:%M:%S")
    )
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False

# Errors we DO retry indefinitely (transient infra failures). Union of OpenAI
# and Anthropic SDK exception classes so both code paths share the same policy.
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
# Errors we do NOT retry (caller's prompt / auth is broken — retrying loops forever).
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

# Exponential backoff with jitter, capped at 60s.
_RETRY_INITIAL_BACKOFF_S = 1.0
_RETRY_MAX_BACKOFF_S = 60.0


DEFAULT_MODELS = {
    "openai": "gpt-4.1-mini",
    "anthropic": "claude-haiku-4-5",
    "umbc": "Claude Haiku 4.5",
}

DEFAULT_BASE_URLS = {
    "openai": None,  # SDK default
    "anthropic": "https://api.anthropic.com/v1/",
    # UMBC's LiteLLM gateway speaks Anthropic Messages API natively; the
    # Anthropic SDK appends /v1/messages so the base_url has no /v1 suffix.
    "umbc": "https://gateway.aws.genai.umbc.edu/",
}

DEFAULT_API_KEY_ENVS = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "umbc": "UMBC_GATEWAY_KEY",
}

# Which SDK to use for each provider. "openai" -> OpenAI SDK (chat completions);
# "anthropic_native" -> Anthropic SDK (messages). Anything in this map is a
# valid `--provider` value.
PROVIDER_SDK = {
    "openai": "openai",
    "anthropic": "openai",      # Anthropic's OpenAI-compatible shim
    "custom": "openai",         # caller's responsibility; SDK is OpenAI-shaped
    "anthropic_native": "anthropic_native",
    "umbc": "anthropic_native",
}

VALID_PROVIDERS = tuple(PROVIDER_SDK.keys())


@dataclass
class ApiConfig:
    provider: str
    model: str
    base_url: Optional[str]
    api_key: str
    temperature: float = 0.0
    max_tokens: int = 16768
    max_concurrency: int = 64
    # Best-effort reproducibility seed forwarded to OpenAI's chat completion
    # endpoint. Anthropic's OpenAI-compatible endpoint does NOT honor `seed`
    # (and may 400 on unknown params), so we skip it for that provider.
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
        raise ValueError(
            f"Unknown provider: {provider}. Valid: {VALID_PROVIDERS}"
        )

    if provider in ("custom", "anthropic_native"):
        if not model or not api_key_env or not base_url:
            raise ValueError(
                f"provider={provider} requires --api_model, --api_key_env, and --api_base_url"
            )
    else:
        # Providers with built-in defaults (openai, anthropic, umbc).
        model = model or DEFAULT_MODELS[provider]
        base_url = base_url or DEFAULT_BASE_URLS[provider]
        api_key_env = api_key_env or DEFAULT_API_KEY_ENVS[provider]

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
    """Issue ONE chat completion, retrying transient failures indefinitely.

    Retry policy:
      - Transient errors (rate limit, timeout, connection, 5xx): retry forever
        with exponential backoff (1s -> 2s -> 4s -> ... capped at 60s) + jitter.
        Each attempt is logged to stderr so it's visible during a long sweep.
      - Permanent errors (bad request, auth, 404, permission): NOT retried — log
        once and return an [API_ERROR] sentinel so the run can complete. These
        indicate a broken prompt or misconfigured key; retrying would loop
        forever.
      - Any other exception: treated as transient (retried). Better to keep
        trying than to silently corrupt a metric with a sentinel row when the
        cause might be a library issue we haven't seen before.
    """
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
                # Pass seed only to providers that honor it (OpenAI + any custom
                # OpenAI-compatible endpoint). Anthropic explicitly does not.
                if cfg.provider != "anthropic":
                    kwargs["seed"] = cfg.seed
                resp = await client.chat.completions.create(**kwargs)
                if attempt > 1:
                    _log.info(
                        "prompt %d: succeeded on attempt %d", prompt_idx, attempt
                    )
                return resp.choices[0].message.content or ""
            except _NON_RETRYABLE_EXCEPTIONS as e:
                _log.error(
                    "prompt %d: NON-RETRYABLE %s on attempt %d: %s — returning sentinel",
                    prompt_idx,
                    type(e).__name__,
                    attempt,
                    str(e)[:200],
                )
                return f"[API_ERROR] {type(e).__name__}: {e}"
            except _RETRYABLE_EXCEPTIONS as e:
                jitter = random.uniform(0, backoff * 0.1)
                sleep_s = min(backoff + jitter, _RETRY_MAX_BACKOFF_S)
                _log.warning(
                    "prompt %d attempt %d transient %s: %s — sleeping %.1fs",
                    prompt_idx,
                    attempt,
                    type(e).__name__,
                    str(e)[:200],
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, _RETRY_MAX_BACKOFF_S)
            except APIError as e:
                # Generic OpenAI APIError that didn't match the specific buckets.
                # Treat as transient.
                jitter = random.uniform(0, backoff * 0.1)
                sleep_s = min(backoff + jitter, _RETRY_MAX_BACKOFF_S)
                _log.warning(
                    "prompt %d attempt %d generic APIError: %s — sleeping %.1fs",
                    prompt_idx,
                    attempt,
                    str(e)[:200],
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, _RETRY_MAX_BACKOFF_S)
            except Exception as e:
                # Unknown error class — treat as transient so a long sweep doesn't
                # die on a single unrecognized failure mode. Logged loudly.
                jitter = random.uniform(0, backoff * 0.1)
                sleep_s = min(backoff + jitter, _RETRY_MAX_BACKOFF_S)
                _log.warning(
                    "prompt %d attempt %d UNKNOWN %s: %s — sleeping %.1fs (treating as transient)",
                    prompt_idx,
                    attempt,
                    type(e).__name__,
                    str(e)[:200],
                    sleep_s,
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
    """Anthropic-Messages-API counterpart of `_one_call`.

    Same retry policy and logging as the OpenAI path. Response shape is
    Anthropic-native: `resp.content[0].text`, not `resp.choices[0].message.content`.
    Used when the upstream proxy is Anthropic-shaped (e.g. UMBC's LiteLLM
    gateway for Claude models).
    """
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
                try:
                    # Anthropic Messages: response.content is a list of content
                    # blocks; the text-bearing block has .type == "text" with a
                    # .text field. Concatenate text blocks (usually just one).
                    parts: List[str] = []
                    for block in resp.content or []:
                        # Tool-use / thinking blocks have no .text — skip them.
                        t = getattr(block, "text", None)
                        if t:
                            parts.append(t)
                    content = "".join(parts)
                except (TypeError, AttributeError, IndexError, KeyError) as parse_err:
                    snippet = str(resp)[:500]
                    _log.error(
                        "prompt %d: NON-RETRYABLE malformed Anthropic response on "
                        "attempt %d (%s). Response head: %s",
                        prompt_idx,
                        attempt,
                        parse_err,
                        snippet,
                    )
                    return f"[API_ERROR] MalformedResponse: {parse_err} | head={snippet}"
                if attempt > 1:
                    _log.info(
                        "prompt %d: succeeded on attempt %d", prompt_idx, attempt
                    )
                return content
            except _NON_RETRYABLE_EXCEPTIONS as e:
                _log.error(
                    "prompt %d: NON-RETRYABLE %s on attempt %d: %s — returning sentinel",
                    prompt_idx,
                    type(e).__name__,
                    attempt,
                    str(e)[:200],
                )
                return f"[API_ERROR] {type(e).__name__}: {e}"
            except _RETRYABLE_EXCEPTIONS as e:
                jitter = random.uniform(0, backoff * 0.1)
                sleep_s = min(backoff + jitter, _RETRY_MAX_BACKOFF_S)
                _log.warning(
                    "prompt %d attempt %d transient %s: %s — sleeping %.1fs",
                    prompt_idx,
                    attempt,
                    type(e).__name__,
                    str(e)[:200],
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, _RETRY_MAX_BACKOFF_S)
            except AnthropicAPIError as e:
                jitter = random.uniform(0, backoff * 0.1)
                sleep_s = min(backoff + jitter, _RETRY_MAX_BACKOFF_S)
                _log.warning(
                    "prompt %d attempt %d generic AnthropicAPIError: %s — sleeping %.1fs",
                    prompt_idx,
                    attempt,
                    str(e)[:200],
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, _RETRY_MAX_BACKOFF_S)
            except Exception as e:
                jitter = random.uniform(0, backoff * 0.1)
                sleep_s = min(backoff + jitter, _RETRY_MAX_BACKOFF_S)
                _log.warning(
                    "prompt %d attempt %d UNKNOWN %s: %s — sleeping %.1fs (treating as transient)",
                    prompt_idx,
                    attempt,
                    type(e).__name__,
                    str(e)[:200],
                    sleep_s,
                )
                await asyncio.sleep(sleep_s)
                backoff = min(backoff * 2, _RETRY_MAX_BACKOFF_S)


async def _run_all(cfg: ApiConfig, prompts: List[str]) -> List[str]:
    sdk_kind = PROVIDER_SDK[cfg.provider]
    sem = asyncio.Semaphore(cfg.max_concurrency)
    if sdk_kind == "anthropic_native":
        client = AsyncAnthropic(api_key=cfg.api_key, base_url=cfg.base_url)
        tasks = [_one_call_anthropic(client, cfg, p, sem, i) for i, p in enumerate(prompts)]
    else:
        client = AsyncOpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        tasks = [_one_call(client, cfg, p, sem, i) for i, p in enumerate(prompts)]
    return await tqdm_asyncio.gather(*tasks, desc=f"API[{cfg.model}]")


def run_api_inference(cfg: ApiConfig, prompts: List[str]) -> List[str]:
    """Blocking entry point: takes a list of prompt strings, returns generations."""
    return asyncio.run(_run_all(cfg, prompts))
