#!/usr/bin/env bash
# Fetch ICEWS05-15 (Garcia-Duran et al. temporal splits, mmkb release) into
# data/icews05-15. Quads are head \t relation \t tail \t YYYY-MM-DD.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/icews05-15 && cd data/icews05-15
if [ -f train.txt ]; then
  echo "data/icews05-15 already present ($(wc -l < train.txt) train quads)"
  exit 0
fi
BASE="https://raw.githubusercontent.com/mniepert/mmkb/master/TemporalKGs/icews05-15"
for split in train valid test; do
  curl -sL -o "$split.txt" "$BASE/icews_2005-2015_$split.txt"
done
echo "fetched: $(wc -l < train.txt) train / $(wc -l < valid.txt) valid / $(wc -l < test.txt) test quads"
echo "next: train with the tranz CLI (see examples/icews14_temporal_clqa.rs header; same recipe, this dataset is ~5x larger)"
