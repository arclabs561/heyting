#!/usr/bin/env bash
# Fetch FB15k-237 (Toutanova & Chen splits, Microsoft release) into data/Release.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data && cd data
if [ -f Release/train.txt ]; then
  echo "data/Release already present ($(wc -l < Release/train.txt) train triples)"
  exit 0
fi
curl -L -o fb15k237.zip \
  "https://download.microsoft.com/download/8/7/0/8700516A-AB3D-4850-B4BB-805C515AECE1/FB15K-237.2.zip"
unzip -o -q fb15k237.zip
echo "fetched: $(wc -l < Release/train.txt) train triples"
echo "next: train embeddings with the tranz CLI (see examples/fb15k237_clqa.rs header)"
