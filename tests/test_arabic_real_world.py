#!/usr/bin/env python3
"""Test real-world Arabic text use cases with natural-pdf."""

import time
import unittest

from natural_pdf import PDF


class TestArabicPDF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        start = time.time()
        cls.pdf = PDF("pdfs/arabic.pdf")
        cls.page = cls.pdf.pages[0]
        cls.load_time = time.time() - start

    def test_load_time_under_3s(self):
        self.assertLess(self.load_time, 3, f"PDF load time too long: {self.load_time:.2f}s")

    def test_text_extraction(self):
        text = self.page.extract_text()
        self.assertTrue(len(text) > 1000, "Extracted text is too short")
        self.assertIn("قانون", text, "Expected Arabic keyword not found in extracted text")

    def test_arabic_keywords_found(self):
        search_terms = [
            ("قانون", "law"),
            ("المنشآت الفندقية", "official gazette"),
            ("رقم", "number"),
            ("لسنة", "for year"),
            ("2022", "year 2022"),
            ("العدد", "issue"),
        ]
        for term, _ in search_terms:
            print("term", term, "translation", _)
            with self.subTest(term=term):
                matches = self.page.find_all(f"text:contains({term})")
                self.assertGreater(len(matches), 0, f"No matches found for term: {term}")

    def test_spatial_navigation(self):
        headers = self.page.find_all("text[size>12]")
        self.assertGreater(len(headers), 0, "No large text headers found")
        below = headers[0].below().find_all("text")
        self.assertGreater(len(below), 0, "No text found below header")

    def test_mixed_content_stats(self):
        words = self.page.find_all("word")
        stats = {
            "arabic_only": 0,
            "english_only": 0,
            "numbers_only": 0,
            "mixed": 0,
        }

        for word in words:
            has_ar = any("\u0600" <= c <= "\u06ff" for c in word.text)
            has_en = any("a" <= c.lower() <= "z" for c in word.text)
            has_num = any(c.isdigit() for c in word.text)

            if has_ar and (has_en or has_num):
                stats["mixed"] += 1
            elif has_ar:
                stats["arabic_only"] += 1
            elif has_en:
                stats["english_only"] += 1
            elif has_num:
                stats["numbers_only"] += 1

        self.assertGreater(stats["arabic_only"], 0, "No Arabic-only words found")
        self.assertGreater(stats["mixed"], 0, "No mixed Arabic/English words found")

    def test_line_grouping_by_y_position(self):
        words = self.page.find_all("word")[:50]
        lines = {}
        for word in words:
            y = round(word.top)
            lines.setdefault(y, []).append(word)
        self.assertGreater(len(lines), 0, "No lines grouped by Y position")

    def test_extraction_consistency(self):
        text1 = self.page.extract_text()
        text2 = self.page.extract_text()
        self.assertEqual(text1, text2, "Text extraction is not consistent")

    def test_bidi_token_order(self):
        found_2022 = self.page.find_all("text:contains(2022)")
        found_2202 = self.page.find_all("text:contains(2202)")
        self.assertGreater(len(found_2022), 0, "Expected to find '2022', but found none")
        self.assertEqual(len(found_2202), 0, "Found reversed '2202', possible BiDi error")
        for w in found_2022:
            print(f"  '2022' sample: '{w.text}' at ({w.x0:.1f}, {w.top:.1f})")
        for w in found_2202:
            print(f"  '2202' sample: '{w.text}' at ({w.x0:.1f}, {w.top:.1f})")


if __name__ == "__main__":
    unittest.main()
