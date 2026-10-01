#### TEST CONFIGURATION FILE ####
# Imports
# Python
from pathlib import Path

# 3rd-party
import pytest

# Gets the code_sample_rag/tests/data directory. This path will work as long as it is called
# by a file within code_sample_rag/tests
@pytest.fixture
def test_data_dir() -> None:
    return Path(__file__).parent / "data"