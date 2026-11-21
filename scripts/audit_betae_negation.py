"""Count lost answers for every published FB15k-237 test 2in and 3in query."""

from collections import defaultdict
from pathlib import Path

from prepare_betae import BenchmarkUnpickler, DATASET


def edges(dataset, splits):
    neighbors = defaultdict(set)
    for split in splits:
        with (dataset / f"{split}.txt").open() as source:
            for line in source:
                head, relation, tail = map(int, line.split())
                neighbors[head, relation].add(tail)
    return neighbors


def read_plain(dataset, stem):
    with (dataset / f"test-{stem}.plain.pkl").open("rb") as source:
        return BenchmarkUnpickler(source).load()


def answers(branches):
    # The last relation is negated in both shapes.
    return set.intersection(*branches[:-1]) - branches[-1]


def main(dataset):
    earlier = edges(dataset, ["train", "valid"])
    added = edges(dataset, ["test"])
    queries = read_plain(dataset, "queries")
    hop = ("e", ("r",))
    negated_hop = ("e", ("r", "n"))
    easy, hard = {}, {}
    for name, shape in [("2in", (hop, negated_hop)), ("3in", (hop, hop, negated_hop))]:
        changed, lost = 0, 0
        for query in queries[shape]:
            keys = [(head, relations[0]) for head, relations in query]
            old = answers([earlier[key] for key in keys])
            new = answers([earlier[key] | added[key] for key in keys])
            removed = old - new
            changed += bool(removed)
            lost += len(removed)
            easy[query] = old
            hard[query] = new - old
        print(f"{name} | queries with losses {changed:,}/{len(queries[shape]):,} | lost pairs {lost:,}")

    del queries, earlier, added
    for kind, expected in [("easy", easy), ("hard", hard)]:
        published = read_plain(dataset, f"{kind}-answers")
        for query, reconstructed in expected.items():
            if published[query] != reconstructed:
                raise ValueError(f"published {kind} answers differ for {query}")
        print(f"{len(expected):,} published {kind} sets matched")
        del published


if __name__ == "__main__":
    main(Path(__file__).resolve().parents[1] / "data" / DATASET)
