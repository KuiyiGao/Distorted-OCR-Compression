import itertools
import math
import unittest
import numpy as np
from deepseek_pipeline import cuad_evaluate, CUADScore
from deepseek_pipeline.qa_eval import squad_em_f1


class LocalEvaluatorTests(unittest.TestCase):
    def test_high_confidence_unanswerable_false_positive_reduces_precision(self):
        score = cuad_evaluate(["wrong", "Delaware"], [[], ["Delaware"]], [.9, .8])
        self.assertEqual(score.precision_at_80_recall, .5)
        self.assertEqual(score.aupr, .25)
        self.assertEqual(score.em, .5)

    def test_tied_confidences_are_permutation_invariant(self):
        cases = [("wrong", [], .8), ("Delaware", ["Delaware"], .8),
                 ("", [], .8)]
        results = []
        for items in itertools.permutations(cases):
            p, g, c = zip(*items)
            score = cuad_evaluate(list(p), list(g), list(c))
            results.append((score.aupr, score.precision_at_80_recall))
        self.assertEqual(set(results), {(.75, .5)})

    def test_single_correct_answer_has_full_area(self):
        score = cuad_evaluate(["Delaware"], [["Delaware"]], [.9])
        self.assertIsInstance(score, CUADScore)
        self.assertEqual(score.aupr, 1)
        self.assertEqual(score.precision_at_80_recall, 1)

    def test_abstention_remains_in_recall_denominator(self):
        score = cuad_evaluate(["Delaware", ""], [["Delaware"], ["New York"]], [.9, 1.0])
        self.assertEqual(score.aupr, .5)
        self.assertEqual(score.precision_at_80_recall, 0)
        self.assertEqual(cuad_evaluate([""], [["Delaware"]]).aupr, 0)

    def test_em_f1_match_existing_qa_reader_for_answerable_and_empty_items(self):
        p = ["the Delaware", "York", "", "wrong"]
        g = [["Delaware", "State of Delaware"], ["New York"], [], []]
        expected = np.mean([squad_em_f1(pred, gold) for pred, gold in zip(p, g)], axis=0)
        score = cuad_evaluate(p, g)
        np.testing.assert_allclose([score.em, score.f1], expected)
        self.assertIn("one tied operating point", score.threshold_note)

    def test_no_answerable_and_empty_batches(self):
        score = cuad_evaluate(["", "wrong"], [[], []])
        self.assertEqual(score.em, .5)
        self.assertTrue(math.isnan(score.aupr))
        empty = cuad_evaluate([], [])
        self.assertEqual(empty.n, 0)
        self.assertTrue(math.isnan(empty.em))

    def test_validation(self):
        for kwargs in ({"confidences": []}, {"confidences": [float("nan")]},
                       {"confidences": [1.1]}, {"jaccard_threshold": 0}):
            with self.assertRaises(ValueError):
                cuad_evaluate(["x"], [["x"]], **kwargs)
        with self.assertRaises(ValueError):
            cuad_evaluate(["x"], [])
        with self.assertRaises(ValueError):
            cuad_evaluate(["x"], [[""]])


if __name__ == "__main__":
    unittest.main()
