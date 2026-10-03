#### INTEGRATION TESTS FOR SERVER ####
# Imports
# Python
import os
from pathlib import Path
from typing import Any

# Codebase
from code_sample_rag import server

# 3rd-party
import pytest
from dotenv import load_dotenv
from fastapi.testclient import TestClient

# Load .env in order to check if GEMINI_API_KEY has been set
load_dotenv()

# Fixture to create TestClient as context manager (so server.lifespan() will run and populate
# app.state.resources for each test that uses client). Setting scope to "module" such that the
# client code only runs once for all tests that use test_client input in this file.
@pytest.fixture(scope="module")
def test_client():
    with TestClient(server.app) as client:
        yield client

# Test for RAGQueryRequest object
def test_rag_query_request():
    just_q_q = "Wei shen me?"
    just_q = server.RAGQueryRequest(
        question=just_q_q
    )
    assert just_q.question == just_q_q
    assert just_q.session_id is None

    all_fields_q = "Y?"
    all_fields_s_id = "meowmeowbark1"
    all_fields = server.RAGQueryRequest(
        question=all_fields_q,
        session_id=all_fields_s_id
    )
    assert all_fields.question == all_fields_q
    assert all_field.session_id == all_fields_s_id

# Test for RAGQueryResponse object
def test_rag_query_response():
    all_fields_a = "Yea lol."
    all_fields_s = [
        {
            "start_page": 25,
            "end_page": 55,
            "chunk": "moo moo moo",
            "file_name": "local_cow.pdf",
            "cos_sim": 0.905
        },
        {
            "start_page": 12,
            "end_page": 21,
            "chunk": "ahwoooo!",
            "file_name": "local_wolf.pdf",
            "cos_sim": 0.83
        }
    ]
    assert all_fields.answer == all_fields_a
    assert all_fields.sources == all_fields_s

# Test for single_turn query
@pytest.mark.integration
@pytest.mark.skipif(
    not os.environ.get("GEMINI_API_KEY"),
    reason="Requires GEMINI_API_KEY to generate embeddings and call LLM"
)
def test_single_turn_query(test_client):
    request_body = {
        "question": "When is Colloquial Levantine Arabic next offered?"
    }
    response = test_client.post(
        "/query",
        json=request_body
    )
    response_json = response.json()
    response_answer = response_json["answer"]

    # Ensure response is successful
    assert response.status_code == 200
    # Expect certain keywords in output
    expected_keywords = {"2017", "spring"}
    assert all(keyword in response_answer.lower() for keyword in expected_keywords), \
           f"Answer '{response_answer}' doesn't contain all expected keywords - '{expected_keywords}'"
    assert len(response_json["sources"]) > 0, f"Expected at least 1 result to be retrieved"

# TODO: Add multi turn test