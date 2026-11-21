"""Prepare the FB15k-237 query files from the Stanford BetaE archive."""

from collections import defaultdict
import hashlib
from pathlib import Path
import pickle
import shutil
import subprocess
from tempfile import TemporaryDirectory
from zipfile import ZipFile


# Observed on 2026-11-21 from the HTTPS endpoint for the archive linked in
# https://github.com/snap-stanford/KGReasoning/tree/ec728497f083973edbe8b9e3c4066f416fc74250
URL = "https://snap.stanford.edu/betae/KG_data.zip"
SHA256 = "85959742a8d525c0026ad95ca154f327649564938142b43c83f1403b6d5d7caf"
DATASET = "FB15k-237-betae"
SPLITS = ("test", "valid")
STEMS = tuple(f"{split}-{kind}" for split in SPLITS
              for kind in ("queries", "easy-answers", "hard-answers"))
RAW = (
    "train.txt", "valid.txt", "test.txt", "stats.txt", "id2ent.pkl", "id2rel.pkl",
    "ent2id.pkl", "rel2id.pkl", "train-queries.pkl", "train-answers.pkl",
    *(f"{stem}.pkl" for stem in STEMS),
)
PLAIN = tuple(f"{stem}.plain.pkl" for stem in STEMS)
RECEIPT = ".prepared-v1"


class BenchmarkUnpickler(pickle.Unpickler):
    """Resolve only the two data constructors used by the pinned pickles."""

    def find_class(self, module, name):
        if (module, name) == ("collections", "defaultdict"):
            return defaultdict
        # Protocol 2 spells the same set constructor __builtin__.set.
        if module in ("builtins", "__builtin__") and name == "set":
            return set
        raise pickle.UnpicklingError(f"unexpected pickle global {module}.{name}")


def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def download(path):
    subprocess.run([
        "curl", "--fail", "--location", "--show-error", "--silent", "--retry", "2",
        "--proto", "=https", "--proto-redir", "=https", "--output", str(path), URL,
    ], check=True)


def prepare(data, expected=SHA256, fetch=download):
    data.mkdir(parents=True, exist_ok=True)
    target = data / DATASET
    receipt = target / RECEIPT
    if (receipt.is_file() and receipt.read_text() == expected
            and all((target / name).is_file() and (target / name).stat().st_size
                    for name in (*RAW, *PLAIN))):
        return

    with TemporaryDirectory(dir=data) as temp:
        stage = Path(temp)
        archive = data / "KG_data.zip"
        if not archive.is_file() or digest(archive) != expected:
            print("Downloading the BetaE archive from Stanford (~1.4 GB).", flush=True)
            part = stage / "download.zip"
            fetch(part)
            if digest(part) != expected:
                raise ValueError("SHA-256 mismatch for Stanford's KG_data.zip")
            part.replace(archive)

        # Copy only the known flat files. Never use archive names as destinations.
        with ZipFile(archive) as bundle:
            for name in RAW:
                with bundle.open(f"{DATASET}/{name}") as source, (stage / name).open("wb") as dest:
                    shutil.copyfileobj(source, dest)

        for stem, plain in zip(STEMS, PLAIN):
            with (stage / f"{stem}.pkl").open("rb") as source:
                values = BenchmarkUnpickler(source).load()
            if not isinstance(values, dict):
                raise ValueError(f"{stem}.pkl should contain a dictionary")
            with (stage / plain).open("wb") as output:
                pickle.dump(dict(values), output, protocol=2)
            del values

        # Keep unrelated user files in the destination. Only complete files
        # become visible, and an interrupted installation has no new receipt.
        target.mkdir(exist_ok=True)
        for name in (*RAW, *PLAIN):
            (stage / name).replace(target / name)
        (stage / RECEIPT).write_text(expected)
        (stage / RECEIPT).replace(receipt)


if __name__ == "__main__":
    prepare(Path(__file__).resolve().parents[1] / "data")
    print("Ready. Run cargo run --release --features tranz --example betae_fb15k237")
