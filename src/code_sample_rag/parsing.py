# Imports
# Python
import re

# 3rd-party
from pypdf import PdfReader

### CONSTANTS ###
SPACE_RE = re.compile(r"[ \t\r]+")

### CLASSES ###
# Just add a wrapper class around pypdf2 extractor object
# to get text from each PDF file location
class PDFParser():
    def __init__(self, input_file: str):
        """
        Basic class for parsing PDF files, breaks text down by pages
        with minimal processing.
        IN:
            input_file - path to input file to parse
        OUT:
            NONE
        """
        self.reader = PdfReader(input_file)

    def parse(self) -> tuple[str, list[int]]:
        """
        Parse text out of the associated PDF file
        IN:
            NONE
        OUT:
            tuple containing parsed text and list of ending indicies in parsed text of page splits
        NOTES:
            TODO: Try Docling or Markitdown if output is terrible
        """
        output_texts = []
        page_indicies = []
        for idx, page in enumerate(self.reader.pages):
            # Play around a bit with the following if needed:
            # https://pypdf.readthedocs.io/en/stable/user/extract-text.html
            page_text = page.extract_text()
            # Normalize non-newline whitespaces
            page_text = SPACE_RE.sub(" ", page_text)
            output_texts.append(page_text)
            page_end_idx = len(page_text)
            # Add offset to culmulative page end index value
            if page_indicies:
                page_end_idx += page_indicies[-1]
            # NOTE: One is added to page indicies to account for extra space added when joining 
            # page content, except for the last page
            if idx < len(self.reader.pages) - 1:
                page_end_idx += 1
            page_indicies.append(page_end_idx)
        # Handle page splits by adding a space
        output_text = " ".join(output_texts)

        return output_text, page_indicies
            