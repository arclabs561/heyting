#!/usr/bin/env bash
# Fetch ICEWS14 (Garcia-Duran et al. temporal splits, mmkb release) into
# data/icews14. Quads are head \t relation \t tail \t YYYY-MM-DD.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/icews14 && cd data/icews14
if [ -f train.txt ]; then
  echo "data/icews14 already present ($(wc -l < train.txt) train quads)"
  exit 0
fi
BASE="https://raw.githubusercontent.com/mniepert/mmkb/master/TemporalKGs/icews14"
for split in train valid test; do
  curl -sL -o "$split.txt" "$BASE/icews_2014_$split.txt"
done
echo "fetched: $(wc -l < train.txt) train / $(wc -l < valid.txt) valid / $(wc -l < test.txt) test quads"
echo "next: train a temporal model with the tranz CLI (see examples/icews14_temporal_clqa.rs header)"
