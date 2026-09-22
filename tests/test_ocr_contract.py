import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from deepseek_pipeline.ocr_compress import DeepSeekOCRCompressor


class OCRContractTests(unittest.TestCase):
    def test_accepts_decoded_text_and_requests_return_value(self):
        for decoded in ("contract text", ["contract text"], ("contract text",)):
            with self.subTest(decoded=decoded), tempfile.TemporaryDirectory() as directory:
                compressor = DeepSeekOCRCompressor.__new__(DeepSeekOCRCompressor)
                compressor.model = MagicMock()
                compressor.model.infer.return_value = decoded
                compressor.tokenizer = object()
                result = compressor.compress("fixture.png", mode="base", output_dir=directory)
                self.assertEqual(result.decoded_text, "contract text")
                self.assertEqual(result.n_vision_tokens, 256)
                self.assertTrue(compressor.model.infer.call_args.kwargs["eval_mode"])
                self.assertFalse(compressor.model.infer.call_args.kwargs["save_results"])

    def test_missing_return_is_not_converted_to_literal_none(self):
        with tempfile.TemporaryDirectory() as directory:
            compressor = DeepSeekOCRCompressor.__new__(DeepSeekOCRCompressor)
            compressor.model = MagicMock()
            compressor.model.infer.return_value = None
            compressor.tokenizer = object()
            with self.assertRaisesRegex(RuntimeError, "did not return decoded text"):
                compressor.compress("fixture.png", output_dir=directory)

    def test_revision_passes_to_both_loaders_without_patching_them(self):
        tokenizer_loader = MagicMock()
        model_loader = MagicMock()
        fake_transformers = SimpleNamespace(AutoTokenizer=tokenizer_loader, AutoModel=model_loader)
        fake_torch = SimpleNamespace(bfloat16=object(), float16=object())
        with patch.dict("sys.modules", {"transformers": fake_transformers, "torch": fake_torch}):
            compressor = DeepSeekOCRCompressor(device="cpu", revision="pinned-revision")
        self.assertEqual(tokenizer_loader.from_pretrained.call_args.kwargs["revision"], "pinned-revision")
        self.assertEqual(model_loader.from_pretrained.call_args.kwargs["revision"], "pinned-revision")
        self.assertIs(compressor.model, model_loader.from_pretrained.return_value)
        self.assertIs(fake_transformers.AutoModel, model_loader)
        compressor.model.eval.assert_called_once()


if __name__ == "__main__":
    unittest.main()
