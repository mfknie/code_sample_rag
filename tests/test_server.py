#### INTEGRATION TESTS FOR SERVER ####
# Imports
# Python
import os
from pathlib import Path
from typing import Any

# Codebase
# Note that server.py import triggers load_dotenv() via setup.py -> models.py,
# adding GEMINI_API_KEY to environment
from code_sample_rag import server

# 3rd-party
import pytest
from fastapi.testclient import TestClient

client = TestClient(server.app)

# Test for single_turn query
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="Requires GEMINI_API_KEY to generate embeddings and call LLM"
)
def test_single_turn_query():
    request_body = {
        "request": "When is Colloquial Levantine Arabic next offered?"
    }
    response = client.post(
        "/query"
    )
    response_json = response.json()
    response_answer = response_json["answer"]

    # Ensure response is successful
    assert response.status_code == 200
    # Expect certain keywords in output
    expected_keywords = {"2017", "spring"}
    assert all(keyword in response_answer for keyword in expected_keywords), \
           f"Answer '{response_answer}' doesn't contain all expected keywords - '{expected_keywords}'"
    assert len(response_json["retrieved_results"]) > 0, f"Expected at least 1 result to be retrieved"

# TODO: Add multi turn test