# Imports
# Python
import json
from pathlib import Path
from typing import Any, Optional

# Codebase
from code_sample_rag import models, ingestion

# 3rd-party
import faiss
from google import genai

# $$$ TODO: Compress and unzip the course catalog docs

### CONSTANTS ###
# Path to config file
CONFIG_PATH = Path(__file__).parent / "app_config.json"

### ERRORS ###
class MissingConfigError(Exception):
    """Raised when app config file is missing."""


### FUNCS ###
def validate_config(config: dict[str, Any]) -> None:
    """
    Validate application configuration file
    IN:
        config: Application config file, expected to exist in <root>/<src>/code_sample_rag/app_config.josn
    OUT:
        NONE (fails if validation fails)
    """
    # TODO: Implement better validation
    assert "chunking" in config
    assert "models" in config
    assert "generate" in config
    assert "retrieval" in config
    pass


def load_genai_resources(config: dict[str, Any]) -> tuple[genai.Client, models.GoogleGenAIEmbed, models.GoogleGenAILLM]:
    """
    Load Google GenAI resources for application usage
    IN:
        config: Application config file, expected to exist in <root>/<src>/code_sample_rag/app_config.json
    OUT:
        Tuple of genai.Client and the embedding model object and LLM object from models.py
    """
    client = models.get_google_genai_client()
    # TODO: Validation for app config file
    em_model_id = config["models"]["EM"]["model_id"]
    llm_model_id = config["models"]["LLM"]["model_id"]
    em = models.GoogleGenAIEmbed(client, em_model_id)
    llm = models.GoogleGenAILLM(client, llm_model_id)
    return client, em, llm


### CLASSES ###
class RAGResources():
    client: genai.Client
    em: models.GoogleGenAIEmbed
    llm: models.GoogleGenAILLM
    index: faiss.IndexFlatIP
    doc_metadata: list[dict[str, Any]]

    def __init__(self):
        # TODO: The prompts/instructions are only meant for the
        #       harvard_2015_2016_bracketed_courses.pdf file
        with open(CONFIG_PATH) as f:
            self.config = json.load(f)
        validate_config(self.config)
        self.client, self.em, self.llm = load_genai_resources(self.config)
        self.index, self.doc_metadata = ingestion.ingest(self.config, self.em)

    def tear_down(self):
        # Persist vector index and metadata
        ingestion.persist_data(self.index, self.doc_metadata)
        # Tear down synchronous client
        self.client.close()
