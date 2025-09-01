"""Test to understand the difference between page.words and page.find_all('text')"""

import pytest

from natural_pdf import PDF


def test_words_vs_find_all_text():
    """Test the difference between page.words and page.find_all('text')"""
    # Use a simple PDF for testing
    pdf = PDF("pdfs/sections.pdf")
    page = pdf.pages[0]

    # Get elements using both methods
    words = page.words
    find_all_text = page.find_all("text")

    # Print types for comparison
    print(f"\npage.words returns: {type(words)}")
    print(f"page.find_all('text') returns: {type(find_all_text)}")

    # Check individual element types
    if words:
        print(f"\nFirst element from page.words: {type(words[0])}")
        print(f"Element attributes: {vars(words[0])}")

    if find_all_text:
        print(f"\nFirst element from page.find_all('text'): {type(find_all_text[0])}")

    # Check element properties
    if words:
        first_word = words[0]
        print(f"\nFirst word type attribute: {getattr(first_word, 'type', 'NO TYPE ATTR')}")
        print(
            f"First word normalized_type: {getattr(first_word, 'normalized_type', 'NO NORMALIZED_TYPE')}"
        )
        print(f"First word text: {first_word.extract_text()}")

    # Count comparison
    print(f"\nCount from page.words: {len(words)}")
    print(f"Count from page.find_all('text'): {len(find_all_text)}")

    # Check if find_all includes chars and words
    chars = page.chars if hasattr(page, "chars") else []
    print(f"\npage.chars count: {len(chars)}")

    # Test the selector filtering logic
    # According to parser.py, 'text' selector matches elements with type in ['text', 'char', 'word']
    # We need to get all elements first
    page._element_mgr.load_elements()
    all_elements = []
    if page._element_mgr._elements:
        for element_list in page._element_mgr._elements.values():
            if isinstance(element_list, list):
                all_elements.extend(element_list)

    all_matching_types = [
        el for el in all_elements if hasattr(el, "type") and el.type in ["text", "char", "word"]
    ]
    print(f"\nTotal elements with type in ['text', 'char', 'word']: {len(all_matching_types)}")

    # Check if any elements have type 'text' vs 'word'
    text_type_count = sum(1 for el in all_matching_types if el.type == "text")
    word_type_count = sum(1 for el in all_matching_types if el.type == "word")
    char_type_count = sum(1 for el in all_matching_types if el.type == "char")

    print(f"\nElements with type='text': {text_type_count}")
    print(f"Elements with type='word': {word_type_count}")
    print(f"Elements with type='char': {char_type_count}")

    # Test exclusions behavior
    print("\n--- Testing Exclusions ---")

    # Add an exclusion to the page
    page.add_exclusion(lambda p: p.find('text:contains("Section 1")'))

    # Get elements again with exclusions
    words_with_exclusions = page.words  # Direct property access
    find_all_with_exclusions = page.find_all("text")  # Uses apply_exclusions=True by default
    find_all_no_exclusions = page.find_all("text", apply_exclusions=False)

    print(f"\nWith exclusion for 'Section 1':")
    print(f"page.words count: {len(words_with_exclusions)}")
    print(f"page.find_all('text') count: {len(find_all_with_exclusions)}")
    print(f"page.find_all('text', apply_exclusions=False) count: {len(find_all_no_exclusions)}")

    # Check what was excluded
    if len(words_with_exclusions) == len(words):
        print("\npage.words does NOT apply exclusions")
    else:
        print(
            f"\npage.words DOES apply exclusions (excluded {len(words) - len(words_with_exclusions)} elements)"
        )


if __name__ == "__main__":
    test_words_vs_find_all_text()
