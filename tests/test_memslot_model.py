import unittest
from types import SimpleNamespace
from unittest.mock import Mock

try:
    import torch
except ModuleNotFoundError:
    torch = None

if torch is not None:
    from deepseek_pipeline.memslot import MemSlotAttention, MemSlotSaliency


@unittest.skipIf(torch is None, "requires the optional torch dependency")
class MemSlotModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(7)
        self.model = MemSlotAttention(d_model=8, n_slots=3, d_proj=4)

    def test_padding_cannot_change_slots_or_valid_reconstruction(self):
        hidden = torch.randn(2, 5, 8)
        mask = torch.tensor([[True, True, True, False, False], [True, False, False, False, False]])
        changed = hidden.clone()
        changed[~mask] += 100
        first = self.model(hidden, mask)
        second = self.model(changed, mask)
        torch.testing.assert_close(first[0][mask], second[0][mask])
        torch.testing.assert_close(first[2], second[2])
        self.assertEqual(float(first[1].masked_select(~mask[:, None, :]).abs().sum().detach()), 0)
        self.assertEqual(float(self.model.saliency(hidden, mask)[~mask].abs().sum().detach()), 0)

    def test_half_precision_padding_has_finite_outputs_and_gradients(self):
        model = self.model.half()
        hidden = torch.randn(1, 5, 8, dtype=torch.float16)
        mask = torch.tensor([[True, True, False, False, False]])
        reconstruction, attention, slots = model(hidden, mask)
        self.assertTrue(all(torch.isfinite(x).all() for x in (reconstruction, attention, slots)))
        reconstruction[mask].float().square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))

    def test_empty_sequences_and_malformed_masks_fail_explicitly(self):
        hidden = torch.randn(2, 5, 8)
        for mask in (torch.zeros(2, 5, dtype=torch.bool), torch.ones(2, 5), torch.ones(2, 4, dtype=torch.bool)):
            with self.subTest(shape=mask.shape, dtype=mask.dtype), self.assertRaises(ValueError):
                self.model(hidden, mask)
        with self.assertRaises(ValueError):
            self.model(torch.empty(1, 0, 8))

    def test_half_precision_constant_saliency_is_finite(self):
        model = self.model.half()
        for hidden, mask in (
            (torch.randn(1, 2, 8, dtype=torch.float16), torch.tensor([[True, False]])),
            (torch.ones(1, 4, 8, dtype=torch.float16), torch.tensor([[True, True, False, False]])),
            (torch.ones(1, 3, 8, dtype=torch.float16), None),
        ):
            with self.subTest(shape=hidden.shape, masked=mask is not None):
                scores = model.saliency(hidden, mask)
                self.assertTrue(torch.isfinite(scores).all())
                torch.testing.assert_close(scores, torch.zeros_like(scores))

    def test_empty_training_is_not_reported_as_a_success(self):
        pipeline = MemSlotSaliency.__new__(MemSlotSaliency)
        pipeline._embed = Mock(return_value=[])
        for contexts in ([], iter(()), "one contract", [""], [None]):
            with self.subTest(contexts=contexts), self.assertRaises(ValueError):
                pipeline.train_on_contracts(contexts, verbose=False)
        pipeline._embed.assert_not_called()
        with self.assertRaisesRegex(ValueError, "no training windows"):
            pipeline.train_on_contracts(["one contract"], verbose=False)

    def test_empty_text_and_invalid_smoothing_do_not_call_the_encoder(self):
        pipeline = MemSlotSaliency.__new__(MemSlotSaliency)
        pipeline.tokenizer = Mock(side_effect=AssertionError("unexpected encoder call"))
        self.assertEqual(pipeline.word_weights("   "), [])
        for sigma in (-1, float("nan"), float("inf"), [1], "2", 1j, True):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                pipeline.word_weights("example", smooth_sigma=sigma)
        pipeline.tokenizer.assert_not_called()

    def test_overlapping_windows_use_character_offsets_and_maximum_scores(self):
        pipeline = MemSlotSaliency.__new__(MemSlotSaliency)
        pipeline.cfg = SimpleNamespace(max_length=3, stride=1)
        pipeline.device = "cpu"
        pipeline.tokenizer = Mock(return_value={
            "input_ids": [[1, 2, 0], [2, 3, 0]],
            "attention_mask": [[1, 1, 0], [1, 1, 0]],
            "offset_mapping": [[(0, 5), (7, 11), (0, 0)], [(7, 11), (12, 17), (0, 0)]],
        })
        pipeline.backbone = Mock(return_value=SimpleNamespace(last_hidden_state=torch.zeros(1, 3, 8)))
        pipeline.memslot = SimpleNamespace(saliency=Mock(side_effect=[
            torch.tensor([[.9, .3, 0.]]), torch.tensor([[.8, .4, 0.]])]))
        weights = pipeline.word_weights("Alpha  βeta\nGamma", smooth_sigma=0)
        self.assertEqual([word for word, _ in weights], ["Alpha", "βeta", "Gamma"])
        torch.testing.assert_close(torch.tensor([value for _, value in weights]), torch.tensor([.9, .8, .4]))


if __name__ == "__main__":
    unittest.main()
