# Imports
# Python
import re
from typing import Any, Optional

CONF_KEY_CHAR_COUNT = "char_count"
CONF_KEY_OVERLAP_CHARS = "overlap_char_count"
# Default config values
# $$$ TODO: Consider removing
DEF_CHAR_COUNT = 200
# Note that overlap chars is optional, defaults to 10 if not provided
DEF_CONF_KEY_OVERLAP_CHARS = 10
DEF_CHUNK_CONFIG = {
    CONF_KEY_CHAR_COUNT: DEF_CHAR_COUNT,
    CONF_KEY_OVERLAP_CHARS: DEF_CONF_KEY_OVERLAP_CHARS
}

# TODO: Breaks on decimals
SENT_END_RE = re.compile(r"[.!?]+")
SPACE_BOUND = re.compile(r"\s+")
RE_CONDS = [SENT_END_RE, SPACE_BOUND]


### FUNCS ###
def validate_config(config: dict[str, Any]=DEF_CHUNK_CONFIG) -> None:
    assert CONF_KEY_CHAR_COUNT in config, "missing char_count"
    assert isinstance(config[CONF_KEY_CHAR_COUNT], int), "char_count is not an integer"
    assert config[CONF_KEY_CHAR_COUNT] > 0, "char_count must be positive"
    # Validation specific to overlap_char_count, which is optional
    if CONF_KEY_OVERLAP_CHARS in config:
        assert isinstance(config[CONF_KEY_OVERLAP_CHARS], int), "overlap_char_count is not an integer"
        assert config[CONF_KEY_OVERLAP_CHARS] > -1, "overlap_char_count must be non-negative"
        assert config[CONF_KEY_OVERLAP_CHARS] < config[CONF_KEY_CHAR_COUNT], "overlap_char_count must be less than char_count"


### CLASSES ###
class CharChunking():
    def __init__(self, config: Optional[dict[str, Any]]=DEF_CHUNK_CONFIG):
        """
        Character chunking class with basic control for
        character count and overlap size
        IN:
            config: dictionary representing JSON configuration for character chunking
        OUT:
            NONE
        """
        validate_config(config)
        self.chunk_char_count = config[CONF_KEY_CHAR_COUNT]
        self.overlap_char_count = config.get(CONF_KEY_OVERLAP_CHARS, DEF_CONF_KEY_OVERLAP_CHARS)
        # Set a floor to avoid chunks that are too small relative to overlap size
        min_chunk_size_ratio = min(self.overlap_char_count/self.chunk_char_count + 0.25, 1)
        self.min_chunk_size = int(min_chunk_size_ratio * self.chunk_char_count)

    def chunk(self, text: str, page_indicies: Optional[list[int]]=None) -> \
              list[tuple[str, int, int]]:
        """
        Create chunks out of input text string
        IN:
            text - input string to chunk
            page_indicies - optional argument containing a list of indicies in text, with each corresponding to where each page ends
        OUT:
            list containing tuple of (<text per page>, <start_page_num>, <end_page_num>)
            If no page_indicies are passed in
        NOTES:
            TODO: Incorporate sentence/word splitting for start of the chunk as well
        """
        output = []
        start_chunk_idx = 0 
        while start_chunk_idx < len(text):
            # Avoid exceeding text length
            end_chunk_idx = min(start_chunk_idx + self.chunk_char_count, len(text) - 1)
            cand_chunk = text[start_chunk_idx: end_chunk_idx + 1]
            # Go through each splitting condition and stop after the first one is reached
            # 1. Splitting on sentence ending characters
            # 2. Splitting on spaces
            # Otherwise, fall back to raw character-count based chunking
            for re_cond in RE_CONDS:
                cond_matches = tuple(re_cond.finditer(cand_chunk))
                if cond_matches:
                    last_occur = cond_matches[-1]
                    # Skip any matches that lead to chunks that are too small
                    if last_occur.end() < self.min_chunk_size:
                        continue
                    cand_chunk = cand_chunk[:last_occur.end()+1]
                    end_chunk_idx = start_chunk_idx + last_occur.end()
                    break
            chunk = cand_chunk
            start_page_num = 1
            end_page_num = 1
            if page_indicies:
                start_page_num, end_page_num = \
                    self.get_page_start_end(start_chunk_idx, end_chunk_idx, page_indicies)
            output.append((chunk, start_page_num, end_page_num))

            # Break if last chunk covers end of text
            if end_chunk_idx >= len(text) - 1:
                break
            
            cand_start_chunk_idx = start_chunk_idx + len(chunk) - self.overlap_char_count
            overlap_char_slice = \
                text[cand_start_chunk_idx: cand_start_chunk_idx + self.overlap_char_count]
            # Use same splitting criteria as above
            for re_cond in RE_CONDS:
                match = re_cond.search(overlap_char_slice)
                if match:
                    cand_start_chunk_idx += match.start()
                    break
            start_chunk_idx = cand_start_chunk_idx

        return output

    def get_page_start_end(self,
                           start_chunk_idx: int,
                           end_chunk_idx: int,
                           page_indicies: list[int]) -> tuple[int, int]:
        """
        Find the page in which a chunk starts and ends
        IN:
            start_chunk_idx - start index of chunk in text
            end_chunk_idx - end index of chunk in text
            page_indicies - a list of indicies in text, with each corresponding to where each page ends
        OUT:
            list containing tuple of (<start_page_num>, <end_page_num>)
        """
        start_page_num = end_page_num = 1
        for idx, page_end_char_idx in enumerate(page_indicies):
            # Convert index into page number
            page_num = idx + 1
            # Take the first page number whose ending index in the text is larger than the chunk's starting
            # index as the starting page
            # TODO: Add check for case where page index of last page is smaller than the total length of the text
            if start_chunk_idx <= page_end_char_idx:
                start_page_num = page_num
                # Similarly, take the last page number whose ending index is smaller than the chunk's ending
                # index as the ending page
                for sub_idx, page_end_char_idx in enumerate(page_indicies[idx:]):
                    end_page_num = page_num + sub_idx
                    if end_chunk_idx <= page_end_char_idx:
                        break
                break

        return start_page_num, end_page_num
