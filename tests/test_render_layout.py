import unittest
from unittest.mock import patch

import numpy as np
try:
    from PIL import ImageDraw
    import matplotlib
except ModuleNotFoundError as error:
    if error.name not in {"PIL", "matplotlib"}:
        raise
    render_dependencies_available = False
else:
    from render.render import render_img, render_img_tiered
    render_dependencies_available = True


@unittest.skipUnless(render_dependencies_available, "Requires optional Pillow and matplotlib dependencies")
class RenderLayoutTests(unittest.TestCase):
    def capture_render(self, renderer, word_weights, **options):
        boxes = []
        original = ImageDraw.ImageDraw.text

        def capture(draw, position, text, *args, **kwargs):
            bounds = draw.textbbox(position, text, font=kwargs.get("font"), anchor=kwargs.get("anchor"))
            boxes.append((text, bounds))
            return original(draw, position, text, *args, **kwargs)

        with patch.object(ImageDraw.ImageDraw, "text", capture):
            image = renderer(word_weights, **options)
        return image, boxes

    def assert_words_visible(self, image, boxes, expected_words):
        self.assertEqual([word for word, _ in boxes], expected_words)
        for word, bounds in boxes:
            with self.subTest(word=word):
                self.assertGreaterEqual(bounds[0], 0)
                self.assertGreaterEqual(bounds[1], 0)
                self.assertLessEqual(bounds[2], image.width)
                self.assertLessEqual(bounds[3], image.height)
                self.assertTrue((np.asarray(image.crop(bounds)) < 255).any())

    def test_continuous_saliency_keeps_all_words_on_narrow_page(self):
        words = [(f"Term{index:02}", index / 39) for index in range(40)]
        image, boxes = self.capture_render(render_img, words, image_width=220)
        self.assert_words_visible(image, boxes, [word for word, _ in words])
        self.assertEqual(image.width, 220)

    def test_tiered_saliency_keeps_all_words_on_narrow_page(self):
        words = [(f"Term{index:02}", index / 39) for index in range(40)]
        image, boxes = self.capture_render(render_img_tiered, words, tiers=[2] * len(words), image_width=220)
        self.assert_words_visible(image, boxes, [word for word, _ in words])
        self.assertEqual(image.width, 220)

    def test_header_and_mixed_tiers_preserve_order(self):
        words = [("first", 0.2), ("second", 0.5), ("third", 0.6), ("last", 1.0)]
        image, boxes = self.capture_render(render_img_tiered, words, tiers=[0, 1, 1, 2], question_text="Fixture", image_width=300)
        self.assert_words_visible(image, boxes, ["Q: Fixture", "first", "second", "third", "last"])

    def test_empty_input_produces_white_placeholder(self):
        for renderer, options in ((render_img, {}), (render_img_tiered, {"tiers": []})):
            image = renderer([], image_width=220, **options)
            self.assertEqual(image.size, (220, 200))
            self.assertTrue((np.asarray(image) == 255).all())

    def test_overwide_word_requests_explicit_layout_change(self):
        words = [("pneumonoultramicroscopicsilicovolcanoconiosis", 0.5)]
        for renderer, options in ((render_img, {}), (render_img_tiered, {"tiers": [2]})):
            with self.subTest(renderer=renderer.__name__), self.assertRaisesRegex(ValueError, "increase image_width or reduce font sizes"):
                renderer(words, image_width=220, **options)

    def test_overwide_header_is_rejected(self):
        words = [("term", 0.5)]
        for renderer, options in ((render_img, {}), (render_img_tiered, {"tiers": [2]})):
            with self.subTest(renderer=renderer.__name__), self.assertRaisesRegex(ValueError, "shorten question_text"):
                renderer(words, question_text="A long header " * 10, image_width=220, **options)

    def test_invalid_width_and_margin_are_rejected(self):
        for renderer, options in ((render_img, {}), (render_img_tiered, {"tiers": []})):
            for width, margin in ((0, 0), (40, 24), (48, 24), (220, -1), (220.5, 24)):
                with self.subTest(renderer=renderer.__name__, width=width, margin=margin), self.assertRaisesRegex(ValueError, "image_width"):
                    renderer([], image_width=width, margin=margin, **options)

    def test_mismatched_or_invalid_tiers_are_rejected(self):
        words = [("first", 0.5), ("second", 0.5)]
        for tiers in ([0], [0, 1, 2], [0, 3], [0, 1.5], [0, 256]):
            with self.subTest(tiers=tiers), self.assertRaises(ValueError):
                render_img_tiered(words, tiers)


if __name__ == "__main__":
    unittest.main()
