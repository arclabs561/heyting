#!/usr/bin/env bash
# Fetch the BetaE/KGReasoning query files (Ren & Leskovec, NeurIPS 2020;
# snap-stanford/KGReasoning) and extract the FB15k-237 subset into
# data/FB15k-237-betae. The archive is ~1.4 GB and carries all three
# datasets; only FB15k-237-betae is extracted.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data && cd data
if [ -f FB15k-237-betae/test-queries.pkl ]; then
  echo "data/FB15k-237-betae already present"
  exit 0
fi
if [ ! -f KG_data.zip ]; then
  echo "downloading KG_data.zip (~1.4 GB)..."
  curl -L -o KG_data.zip "http://snap.stanford.edu/betae/KG_data.zip"
fi
unzip -o -q KG_data.zip 'FB15k-237-betae/*'
# The query/answer pickles are Python defaultdicts (a class reference the
# Rust pickle reader cannot materialize); convert once to plain dicts.
python3 - <<'PYEOF'
import pickle
base = "FB15k-237-betae"
for stem in ["test-queries", "test-easy-answers", "test-hard-answers",
             "valid-queries", "valid-easy-answers", "valid-hard-answers"]:
    with open(f"{base}/{stem}.pkl", "rb") as f:
        d = pickle.load(f)
    with open(f"{base}/{stem}.plain.pkl", "wb") as f:
        pickle.dump(dict(d), f, protocol=2)
print("converted 6 pickles to plain dicts")
PYEOF
echo "extracted: $(find FB15k-237-betae -type f | wc -l) files in data/FB15k-237-betae"
echo "next: cargo run --release --features tranz --example betae_fb15k237"
