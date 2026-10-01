# Python
import http
import os
import time
import traceback
from typing import Any, Optional, Union

# 3rd-party
import numpy as np
from dotenv import load_dotenv, dotenv_values
# See https://ai.google.dev/gemini-api/docs for more details
# NOTE: Development was done with free tier
from google import genai

# Add GEMINI_API_KEY to environment from .env
load_dotenv()
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

### CONSTANTS ###

#### EMBED ####
# TODO: Make this configurable
MODEL_ID_GEMINI_EMBED_001 = "gemini-embedding-001"
# "gemini-embedding-001" task types, added only relevant tasks
# See https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-1
TASK_TYPE_001_RET_DOC = "RETRIEVAL_DOCUMENT"
TASK_TYPE_001_RET_QUERY = "RETRIEVAL_QUERY"
# Use smaller dimensionality to save space - 
# See https://ai.google.dev/gemini-api/docs/embeddings#control-embedding-size
EMBED_DIM_001 = 768
# TODO: Add task instructions for "gemini-embedding-2"
# See https://ai.google.dev/gemini-api/docs/embeddings#task-types-embeddings-2

# Based on google.genai.errors.ClientError: 400 INVALID_ARGUMENT. {'error': {'code': 400, 'message': '* BatchEmbedContentsRequest.requests: at most 100 requests can be in one batch\n', 'status': 'INVALID_ARGUMENT'}}
BATCH_SIZE = 25
# Seconds to wait before moving to the next batch
BATCH_TIME_DELAY = 65

# Want to get the following JSON from err_opject.details.error.details array
# {'@type': 'type.googleapis.com/google.rpc.RetryInfo', 'retryDelay': '42s'}
RETRY_INFO_SUFFIX = "RetryInfo"
AT_TYPE_FIELD = "@type"
RETRY_DELAY_FIELD = "retryDelay"

# Number of additional seconds to wait beyond the given retryDelay value
RETRY_DELAY_BUFFER = 2

# Number of retries per embedding call
MAX_RETRIES = 10

#### LLM ####
MODEL_ID_GEMINI_3_1_FLASH_LITE = "gemini-3.1-flash-lite"

### ERRORS ###
class UnsupportedEmbedModelError(Exception):
    """Raised when input embedding model is not supported."""

class NoMoreRetries(Exception):
    """Raised when no more retries are left"""


### FUNCS ###
def get_google_genai_client():
    """
    Given GEMINI_API_KEY in environment, spins up google.genai.Client
    IN:
        NONE
    OUT:
        genai.Client object
    """
    assert os.getenv("GEMINI_API_KEY"), f"GEMINI_API_KEY env var must be set"
    client = genai.Client()
    return client

def extract_retry_delay(err: genai.errors.ClientError) -> Union[float, None]:
    """
    IN:
        err: genai.errors.ClientError
    OUT:
        bool representing whether the error is a 429 error due to resource
        exhaustion
    """
    retry_value = None
    if err.status == "RESOURCE_EXHAUSTED":
        try:
            error_details_list = err.details["error"]["details"]
            for details_dict in error_details_list:
                if details_dict[AT_TYPE_FIELD].endswith(RETRY_INFO_SUFFIX):
                    raw_delay = details_dict[RETRY_DELAY_FIELD].rstrip("s")
                    retry_value = float(raw_delay)
        except Exception as e:
            print(f"Hit the following error while trying to extract retry delay, skipping: '{e}'")
    else:
        print(f"The following is not a 429 (RESOURCE EXHAUSTED) error, skipping retry delay extraction: '{err}'")

    return retry_value


### CLASSES ###
class GoogleGenAIEmbed():
    def __init__(self, client: genai.Client, model_id: str):
        """
        Class for embedding model calls from google.genai
        See https://ai.google.dev/gemini-api/docs/embeddings for more details
        IN:
            client: Google GenAI client needed to connect to embedding models
            model_id: Embedding model model id to use
        OUT:
            NONE
        """
        self.client = client
        if model_id != MODEL_ID_GEMINI_EMBED_001:
            raise UnsupportedEmbedModelError(f"Embedding model '{model_id}' is not supported currently")
        self.model_id = model_id

    def embed_text(self, texts: list[str], task_type_str: str) -> np.ndarray:
        """
        Embed a given list of texts
        IN:
            texts: list of input strings to embed
            task_type_str: the embedding task to perform as a string (this field
                           may have different meanings based on the model_id)
        OUT:
            Numpy array of the shape (<text_count>, <embed_dim>), which each row representing an embedding
            generated from input texts
        """
        other_kwargs = {}
        # TODO: Support other embedding models
        if self.model_id == MODEL_ID_GEMINI_EMBED_001:
            other_kwargs["config"] = genai.types.EmbedContentConfig(
                task_type=task_type_str,
                output_dimensionality=EMBED_DIM_001
            )

        # List of resulting embeddings
        results = []
        # Batch results, as embed_content() only takes 100 entries in a call at most
        text_batches = [texts[idx: idx + BATCH_SIZE] for idx in range(0, len(texts), BATCH_SIZE)]
        
        for batch_num, batch in enumerate(text_batches):
            print(f"Now sending {len(batch)} items for batch number {batch_num}")
            obtained_batch = False
            for retry_count in range(MAX_RETRIES):
                try:
                    batch_result = self.client.models.embed_content(
                        model=self.model_id,
                        contents=batch,
                        **other_kwargs
                    )
                    obtained_batch = True
                    break
                except genai.errors.ClientError as e:
                    # If 429 resource exhausted error pops up, wait 50 seconds for the next request
                    # In practice, I saw about 42-47 seconds of delay 
                    retry_delay = extract_retry_delay(e)
                    if retry_delay:
                        print(f"Hit 429, sleeping for {retry_delay} secs, on retry attempt {retry_count + 1}")
                        time.sleep(retry_delay + RETRY_DELAY_BUFFER)
                    else:
                        raise e
            
            if not obtained_batch:
                raise NoMoreRetries(f"Ran out of retries after {MAX_RETRIES} attempts for model {self.model_id}")

            results.extend(batch_result.embeddings)
            # Sleep some seconds before moving on to next batch, except for last batch
            if batch_num < len(text_batches) - 1:
                print(f"Sleeping {BATCH_TIME_DELAY} before next batch - ZZZ!")
                time.sleep(BATCH_TIME_DELAY)
        
        embeddings = np.asarray([embedding.values for embedding in results], dtype=np.float32)
        return embeddings


class GoogleGenAILLM():
    def __init__(self, client: genai.Client, model_id: str=MODEL_ID_GEMINI_3_1_FLASH_LITE):
        """
        Class for LLM calls from google.genai
        See https://ai.google.dev/gemini-api/docs/embeddings for more details
        IN:
            client: Google GenAI client needed to connect to embedding models
            model_id: LLM model id to use
        OUT:
            NONE
        """
        self.client = client
        self.model_id = model_id

    def generate_text(self,
                      text: str,
                      system_instruction: str="",
                      generation_config: dict[str, Any]={},
                      prev_inter_id: Optional[str]=None) -> tuple[str, str]:
        """
        Call the given LLM for the given text and other parameters
        See https://ai.google.dev/gemini-api/docs/text-generation for more details
        IN:
            text: Query to send to LLM
            system_instruction: System prompt to use for call
            generation_config: Parameters for LLM call (see Google GenAI documentation)
            prev_inter_id: ID of prior interaction call (for multi-turn)
        OUT:
            tuple of the form (<llm_response>, <interaction_id>)
        """
        other_kwargs = {}
        if system_instruction:
            other_kwargs["system_instruction"] = system_instruction
        if generation_config:
            other_kwargs["generation_config"] = generation_config
        if prev_inter_id:
            other_kwargs["previous_interaction_id"] = prev_inter_id
        # TODO: Add all relevant paramters in https://ai.google.dev/api/interactions-api
        result = self.client.interactions.create(
            model=self.model_id,
            input=text,
            **other_kwargs
        )
        return result.output_text, result.id
