from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import pickle
from tempfile import TemporaryDirectory
import unittest

from audit_betae_negation import main


class AuditTests(unittest.TestCase):
    def test_reconstructs_both_shapes_and_detects_wrong_labels(self):
        with TemporaryDirectory() as temp:
            data = Path(temp)
            # Both targets meet the extra positive condition of 3in. A test
            # edge rules out 1; a validation edge keeps 2 eligible for 3in.
            (data / "train.txt").write_text("0 0 1\n0 2 1\n")
            (data / "valid.txt").write_text("0 2 2\n")
            (data / "test.txt").write_text("0 1 1\n0 0 2\n")
            positive = ("e", ("r",))
            negated = ("e", ("r", "n"))
            q2 = ((0, (0,)), (0, (1, -2)))
            q3 = (q2[0], (0, (2,)), q2[1])
            types = {(positive, negated): {q2}, (positive, positive, negated): {q3}}
            for stem, obj in [("queries", types), ("easy-answers", {q2: {1}, q3: {1}}),
                              ("hard-answers", {q2: {2}, q3: {2}})]:
                (data / f"test-{stem}.plain.pkl").write_bytes(pickle.dumps(obj, protocol=2))
            output = StringIO()
            with redirect_stdout(output):
                main(data)
            for name in ("2in", "3in"):
                self.assertIn(f"{name} | queries with losses 1/1 | lost pairs 1", output.getvalue())
            (data / "test-hard-answers.plain.pkl").write_bytes(pickle.dumps({q2: {1}, q3: {2}}))
            with self.assertRaisesRegex(ValueError, "published hard answers differ"), redirect_stdout(output):
                main(data)
