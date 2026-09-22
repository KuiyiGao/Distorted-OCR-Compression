import unittest

from deepseek_pipeline import SaliencyPruner


class WhitespaceTokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()


class SaliencyPrunerTests(unittest.TestCase):
    def setUp(self):
        self.pruner = SaliencyPruner(WhitespaceTokenizer())

    def test_ties_do_not_exceed_floor_word_budget(self):
        result = self.pruner.compress([("one", .9), ("two", .9), ("three", .9), ("four", .1)], .5)
        self.assertEqual(result.text, "one two")
        self.assertEqual(result.n_tokens, 2)

    def test_selected_words_remain_in_reading_order(self):
        result = self.pruner.compress([("one", .1), ("two", .8), ("three", .9), ("four", .2)], .5)
        self.assertEqual(result.text, "two three")

    def test_zero_full_and_empty_budgets(self):
        words = [("one", .1), ("two", .8)]
        for ratio in (0, .1):
            self.assertEqual(self.pruner.compress(words, ratio).text, "")
        self.assertEqual(self.pruner.compress(words, 1).text, "one two")
        self.assertEqual(self.pruner.compress([], .5).n_tokens, 0)

    def test_invalid_ratios_and_weights(self):
        for ratio in (-.1, 1.01, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                self.pruner.compress([("word", .5)], ratio)
        for words in ([('word', float('nan'))], [('word', float('inf'))], [(' ', .5)]):
            with self.assertRaises(ValueError):
                self.pruner.compress(words, .5)


if __name__ == "__main__":
    unittest.main()
