# Summary
A basic RAG system using Google GenAI and FAISS for AI backend and FastAPI for the frontend that can answer questions based on the 2015-2016 bracketed courses catalog from Harvard (https://registrar.fas.harvard.edu/sites/g/files/omnuum1531/files/fasro/files/fas_bracketed_courses_2015-16.pdf)

# Quickstart

1. Clone this repository
2. Install `uv`
3. Run `uv sync` in the root directory of this repository (with `pyproject.toml` and `uv.lock`)
4. Add `GEMINI_API_KEY` to `.env` (see `.env.example` for format)
5. Run `uv run uvicorn code_sample_rag.server:app` to begin the server. This setup step can take up to 10 minutes due to Google GenAI free tier limits/throttling.
6. TODO: How to call the API/use front-end

# Details
## Data
I am using the Harvard 2015-2016 Bracketed Course catalog (approximately 216 pages), as the size of the
document allows for relatively fast setup for this code sample.
The system will technically ingest any .pdf files placed into /data, but for simplicity, this is the only document included. Further prompting changes may be required depending on the data set and the intended
usage of the agent.

Fixed size chunking and PDF parsing are implememented here as a start.

## Configuration
The app_config.json file controls configuration options across the application, including parameters
related to chunking, parsing, LLM/embedding models, prompting and retrieved results.

## LLM/Embedding models
This app uses Gemini models, and as such expects the GEMINI_API_KEY environment variable to be set prior to startup (the code sample expects a .env file in the root directory of the project).

The following models are used in the current configuration:
* LLM - gemini-3.1-flash-lite
* Embedding model - gemini-embedding-001
The models were primarily chosen due to being the least expensive models still supported (see pricing details in https://ai.google.dev/gemini-api/docs/pricing). Note that code will need to be modified to use `gemini-embedding-2`, as the embedding model doesn't support the "task_type" parameter. Due to relatively small sized data set, I didn't consider any quantization.

Prompting config contains customizable components for the system prompt, context instructions and formatting, and the separator value between different components.

## Vector index
A flat, local FAISS index is used to store embeddings for semantic search, as the included dataset is 
relatively small (approximately 230 vectors after chunking). As such, using indexes (e.g. IVF, HNSW) or quantization would not lead to much benefit in speed or memory usage. In larger applications, it would
make sense to use a dedicated database (or other solution) with vector functionality.

## Tests
Unit tests can be run with the following command from anywhere in the project
```
uv run pytest
```
Integration tests are present in tests/

# Future improvements
Improvements within each category are listed in rough order of importance.
*Short-term TODOs*
- Validate API and LLM call
- Create static frontend with FastAPI after API
## General
- Add actual tracing/logs (probably from OpenTelemetry)
## Ingestion
- Expand and improve parsing (could use out-of-box solution like Markitdown, Docling, etc...)
    - Add parsing support for more documetn types
- Multilingual parsing/chunking (e.g. chunking assumes English punctuation)
- Add more chunking strategies (e.g. document structure chunking (e.g. using <h[1-6]> tags in HTML files), contextual chunking (e.g. attaching LLM-generated summary of surrounding context), semantic chunking (using sentence embeddings to determine related adjacentcontent))
- Post-processing to add more detail/context for chunks
## LLM/embedding models
- Use more secure cloud access methods (e.g. service accounts, instance principals, workload identity federation)
- Support for more LLM/embedding model providers and models
- Multimodal model support
- Concurrency for RAG calls (if multiple users are ever expected)
## RAG
0 Add configurable guardrails layer based on user provided keywords/regex or Guardrails APIs/models
## Frontend
- Use Svelte/React to build a more complete frontend
- Frontend changes that require additional models.py, generate.py or retrieval.py changes
    - Enable streaming
    - Enable editing of prior messages and allow conversation branching
    - Save and resume chats from history
    - Improve semantic search to incorporate prior model replies and user questions over the conversation

# AI Usage
AI was used for code review and for technical questions.