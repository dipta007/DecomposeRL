set -e -x

read -p "This will delete all data/combined/step_* files. Continue? (y/N) " confirm
if [[ "$confirm" != "y" && "$confirm" != "Y" ]]; then
    echo "Aborted."
    exit 1
fi

DIR=decomposer/data_process
LOG=data/combined/logs

rm -rf data/combined/step_*
rm -rf data/combined/plots
mkdir -p "$LOG"

run() { echo "$1"; PYTHONPATH=. uv run "$DIR/$2" | tee "$LOG/$3"; }

# Step 0: Collect raw data
run "Get all data..."                             get_all.py                     step_0.log

# Step 1: Rule-based filtering
run "Filter based on rules..."                    filter_rule_based.py           step_1.log
run "Analysis filter..."                          analysis_filter_thresholds.py  analysis_filter_thresholds.log

# Step 2: NER-based filtering
run "Extract NER..."                              ner_claims.py                  ner_extraction.log
run "Filter based on NER..."                      filter_num_of_ner.py           step_2.log
run "Analyze NER complexity..."                   analysis_ner_thresholds.py     analysis_ner_thresholds.log

# Step 3: Confidence-based filtering
run "Filter via MiniChecker..."                   filter_confidence_complexity.py step_3.log

# Step 4: Semantic similarity filtering
run "Embed claims..."                             embed_claims.py                embedding_extraction.log
run "Filter by semantic similarity..."            filter_semantic.py             step_4.log

# Step 5: Decompose claims
run "Decompose claims via GPT..."                 decompose_claims.py            step_5.log
run "Analyze decompositions..."                   analysis_decompositions.py     analysis_decompositions.log

# Step 6: Filter by decomposition count
run "Filter by num decompositions..."             filter_num_decomps.py          step_6.log

# Step 7: Diversity selection (pick one or compare all)
run "Diversify data (facility location)..."       diversify_submod.py            step_7_submod.log
run "Diversify data (random)..."                  diversify_random.py            step_7_random.log
run "Diversify data (kmeans)..."                  diversify_kmeans.py            step_7_kmeans.log
run "Diversify data (farthest point)..."          diversify_farthest.py          step_7_farthest.log
run "Diversify data (kmeans+dpp)..."              diversify_kmeans_dpp.py        step_7_kmeans_dpp.log
run "Compare diversity methods..."                compare_diversity.py           analysis_diversity.log

# Step 8: Augment with long-evidence samples
run "Augment with long evidence..."              augment_long_evidence.py       step_8.log

# Step 9: Decompose claims for augmented samples
run "Decompose augmented claims..."              decompose_augmented.py         step_9.log
