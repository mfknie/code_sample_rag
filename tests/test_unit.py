#### UNIT TESTS ####
# Imports
# Python
import os
from pathlib import Path
from typing import Any

# Codebase
from code_sample_rag import parsing, chunking

# 3rd-party
import pytest

# Test parsing functionality on test PDF file
def test_parsing(test_data_dir: Path):
    test_pdf_file = test_data_dir / "oracle_fy27_press.pdf"
    parser = parsing.PDFParser(test_pdf_file)
    try:
        parsed_text, page_end_indicies = parser.parse()
    except Exception as e:
        assert False, f"While parsing {test_pdf_file}, hit exception: '{e}'"

    assert isinstance(parsed_text, str)
    assert len(parsed_text) > 1000, f"Expected text to be over 1000 chars long, got {len(parsed_text)} chars instead"
    assert len(page_end_indicies) == 11, \
        f"Expected page end indicies for 11 pages to be extracted, got {len(page_end_indicies)} instead"
    print(page_end_indicies)
    assert page_end_indicies[-1] == len(parsed_text), \
        f"Last page end index is expected to be length of text - {len(parsed_text)}, but got {page_end_indicies[-1]} instead"

# Chunking initialization cases that should hit validation errors
@pytest.mark.parametrize(
    "input_config", 
    [
        # Missing "char_count"
        ({"random_key": 1}),
        # "char_count" not int
        ({"char_count": "cat"}),
        # Non-positive "char_count"
        ({"char_count": 0}),
        # "overlap_char_count" not int
        ({"char_count": 5, "overlap_char_count": ["dog", "bunny"]}),
        # Negative "overlap_char_count"
        ({"char_count": 5, "overlap_char_count": -4}),
        # "overlap_char_count" > "char_count"
        ({"char_count": 5, "overlap_char_count": 500})
    ]
)
def test_chunking_config_validation(input_config: dict[str, Any]):
    with pytest.raises(AssertionError):
        chunking.CharChunking(input_config)

# Chunking cases that should pass
# 138 chars
multi_word_text = (
    "Bulbasaur Ivysaur Venusaur Charmander Charmeleon Charizard Squirtle Wartortle Blastoise "
    "Caterpie Metapod Butterfree Weedle Kakuna Beedrill"
)
multi_word_text_chunks = [
    ("Bulbasaur Ivysaur V", 1, 1),
    (" Venusaur Charmander ", 1, 2),
    (" Charmeleon C", 2, 2),
    (" Charizard Squirtle W", 2, 3),
    (" Wartortle Blastoise ", 3, 4),
    (" Caterpie Metapod B", 4, 4),
    (" Butterfree Weedle K", 4, 5),
    (" Kakuna B", 5, 5),
    (" Beedrill", 5, 5),
]
# 390 chars, 126 words, modified to add extra question mark
multi_sentence_text = (
    "I’ll admit that you’re stronger than before. But because of that, "
    "you seem to have lost sight of what’s truly important. Now listen carefully! " 
    "The people of the village, who once abhorred you, have begun to admire you. "
    "They think of you as a comrade? It’s because you fought hard for their acceptance. "
    "You once said it was everyone who cares about you who helped you get where you are now. "
)
multi_sentence_text_chunks = [
    ("I’ll admit that y", 1, 1),
    (" you’re stronger t", 1, 1),
    (" than before. ", 1, 1),
    (". But because of t", 1, 1),
    (" that, you seem to h", 1, 1),
    (" have lost sight of w", 1, 1),
    (" what’s truly i", 1, 1),
    (" important. ", 1, 1),
    (". Now listen c", 1, 1),
    (" carefully! ", 1, 2),
    ("! The people of the v", 1, 2),
    (" village, who once a", 2, 2),
    (" abhorred you, have b", 2, 2),
    (" begun to admire you.", 2, 2),
    (". They think of you a", 2, 2),
    (" as a comrade? ", 2, 2),
    ("? It’s because you f", 2, 2),
    (" fought hard for t", 2, 2),
    (" their acceptance. ", 2, 2),
    (". You once said it w", 2, 2),
    (" was everyone who c", 2, 2),
    (" cares about you who ", 2, 2),
    (" helped you get w", 2, 2),
    (" where you are now. ", 2, 2)
]

@pytest.mark.parametrize(
    "input,expected", 
    [
        # One-word input text with 1 page
        ({"text": "a" * 50, "page_indicies": [49]},
         [("aaaaaaaaaaaaaaaaaaaaa", 1, 1),
          ("aaaaaaaaaaaaaaaaaaaaa", 1, 1),
          ("aaaaaaaaaaaa", 1, 1)]),
        # One-word input text with several pages, with at least
        # one chunk spanning multiple pages
        ({"text": "b" * 50, "page_indicies": [6, 7, 15, 49]},
         [("bbbbbbbbbbbbbbbbbbbbb", 1, 4),
          ("bbbbbbbbbbbbbbbbbbbbb", 4, 4),
          ("bbbbbbbbbbbb", 4, 4)]),
        # Multi-word test
        ({"text": multi_word_text, "page_indicies": [25, 57, 86, 115, 137]}, 
         multi_word_text_chunks), 
        # Multi-sentence test
        ({"text": multi_sentence_text, "page_indicies": [141, 389]},
         multi_sentence_text_chunks),
        # Problematic test case, matching condition 
         ({"text": "c." + " " * 30 + "d", "page_indicies": [10, 32]},
         [("c.                   ", 1, 2),
          ("             d", 2, 2)])
    ]
)
def test_chunking_pass(input: dict[str, Any], expected: list[tuple[str, int, int]]):
    # Use smaller sized config for simplicity
    char_chunking = chunking.CharChunking(
        {
            chunking.CONF_KEY_CHAR_COUNT: 20,
            chunking.CONF_KEY_OVERLAP_CHARS: 2
        }
    )
    chunk_out = char_chunking.chunk(**input)
    assert chunk_out == expected, f"Got unexpected chunk output:{chunk_out}"
