"""Evaluate a witnessed, profession-constrained two-hop query on FB15k-237."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

DIGEST = {
    "train": "6e4c2782169af21e9743f3b1d200886f5d595bf6bc504ec1351720949c5cdfae",
    "valid": "cf6309010852f6a8d47a45df830a426415d1ee6f7a3970a8376ff1fb81db4a5c",
    "test": "5711cf41623ceb4eacc50eb6108a3ca6565c7492e3caaf82a3e355cc660d1574",
}
DIRECTED = "/film/director/film"
ACTED = "/film/actor/film./film/performance/film"
PROFESSION = "/people/person/profession"


class Graph:
    def __init__(self):
        self.films = defaultdict(set)
        self.cast = defaultdict(set)
        self.by_profession = defaultdict(set)

    def add(self, triples):
        for h, r, t in triples:
            if r == DIRECTED:
                self.films[h].add(t)
            elif r == ACTED:
                self.cast[t].add(h)
            elif r == PROFESSION:
                self.by_profession[t].add(h)

    def actors(self, director, limit=None):
        films = self.films.get(director, set())
        by_cast = sorted(films, key=lambda f: (-len(self.cast.get(f, set())), f))
        return set().union(*(self.cast.get(f, set()) for f in by_cast[:limit]))


def read_split(folder, name):
    raw = (folder / f"{name}.txt").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGEST[name]:
        raise ValueError("Corpus digest mismatch: " + name)
    return [line.split("\t") for line in raw.decode().splitlines()]


def evaluate(folder, verification=None):
    train, dev, full = Graph(), Graph(), Graph()
    for name, graphs in (("train", (train, dev, full)), ("valid", (dev, full)), ("test", (full,))):
        triples = read_split(folder, name)
        for g in graphs:
            g.add(triples)
    actors = set().union(*train.cast.values())
    professions = sorted(train.by_profession,
                         key=lambda p: (-len(actors & train.by_profession[p]), p))[:8]
    if verification is not None:
        verification.mkdir(parents=True, exist_ok=True)
        edges = [(director, 0, film) for director, films in train.films.items() for film in films]
        edges.extend((film, 1, actor) for film, cast in train.cast.items() for actor in cast)
        edges.extend((p, 2, actor) for p, members in train.by_profession.items() for actor in members)
        ids = {v: i for i, v in enumerate(sorted({v for h, _, t in edges for v in (h, t)}))}
        with (verification / "edges.tsv").open("w") as out:
            print(len(ids), file=out)
            for h, r, t in sorted(edges):
                print(ids[h], r, ids[t], sep="\t", file=out)
    expected = []
    counts = Counter()
    no_hit = Counter()
    example = None
    for d in sorted(train.films):
        observed, with_dev, all_actors = (g.actors(d) for g in (train, dev, full))
        limited = {k: train.actors(d, k) for k in (1, 3, 8)}
        for p in professions:
            counts["queries"] += 1
            known = observed & train.by_profession[p]
            if verification is not None:
                expected.append([ids[d], ids[p], *sorted(ids[a] for a in known)])
            validation = with_dev & dev.by_profession[p]
            reference = all_actors & full.by_profession[p]
            if not known <= validation <= reference:
                raise AssertionError("Monotonicity failed")
            counts["unfiltered_two_hop_candidates"] += len(observed)
            counts["full_snapshot_answers"] += len(reference)
            counts["train_witnessed"] += len(known)
            counts["new_validation_witness"] += len(validation - known)
            counts["new_test_witness"] += len(reference - validation)
            counts["queries_with_any_train_answer"] += bool(known)
            counts["queries_with_any_full_answer"] += bool(reference)
            for k, cast in limited.items():
                found = cast & train.by_profession[p]
                counts[f"at_{k}_films_answers"] += len(found)
                no_hit[str(k)] += bool(known and not found)
            if example is None and known:
                actor = min(known)
                film = min(f for f in train.films[d] if actor in train.cast.get(f, set()))
                example = [[d, DIRECTED, film], [actor, ACTED, film], [actor, PROFESSION, p]]
    if verification is not None:
        with (verification / "expected.tsv").open("w") as out:
            for row in expected:
                print(*row, sep="\t", file=out)
    return dict(sha256=DIGEST, query_directors=len(train.films), profession_ids=professions,
                counts=counts, positive_queries_emptied_by_pruning=no_hit, example_witness=example)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.corpus, args.verification), indent=2, sort_keys=True))
