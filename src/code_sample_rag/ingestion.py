# Imports
# Python
import os
import pickle
from pathlib import Path
from typing import Any

# Codebase
from code_sample_rag import models, parsing, chunking

# 3rd party
import faiss
import numpy as np
from google import genai

### CONSTANTS ###
# This file(ingestion.py)'s path is <root>/src/code_sample_rag/ingestion.py
INDEX_CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "stored_embeddings.faiss"
TEMP_INDEX_CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "temp_stored_embeddings.faiss"
METADATA_CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "stored_metadata.pickle"
TEMP_METADATA_CACHE_PATH = Path(__file__).resolve().parents[2] / "cache" / "temp_stored_metadata.pickle"
DATA_PATH = Path(__file__).resolve().parents[2] / "data"
BRACKET_COURSE_CATALOG_FILE_PATH = DATA_PATH / "harvard_2015_2016_bracketed_courses.pdf"

# Access existing vector store, store set of documents, set up vector store with flat index - all in memory
# Explain that index is not required for small document base (~10000 chunks max), point to https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index

### FUNCS ###
# TODO: Add to run CRUD on existing FAISS index
# TODO: Add support for other indexes (e.g. IVF, HNSW)
def create_vec_idx(data: np.ndarray) -> faiss.IndexFlatIP:
    """
    Set up vector index given text data
    IN:
        data: Input numpy array with shape (<cow>, <dog>)
    OUT:
        A flat inner-product FAISS index (see https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)
    """
    index = faiss.IndexFlatIP(models.EMBED_DIM_001)
    # Normalize vectors
    data_norm = np.linalg.norm(data, axis=1, keepdims=1)
    # NOTE: Probably don't need to avoid division by 0, as client.models.embed_content() would need to return a vector
    # with only 0s
    data = data/data_norm
    index.add(data)
    return index

def persist_data(index: faiss.IndexFlatIP, doc_metadata: list[dict[str, Any]]) -> None:
    """
    Persist a FAISS vector index and metadata
    IN:
        index: An existing flat inner-product FAISS index
               (see https://github.com/facebookresearch/faiss/wiki/Faiss-indexes)\
        doc_metadata: Existing metadata per chunk, must be in the same order as the embeddings that were
                      inserted into index
                      Each metadata JSON is of the form
                      { 
                          "start_page": <starting_page_num>,
                          "end_page": <ending_page_num>,
                          "chunk": <chunk_content>,
                          "file_name": <file_name>
                      }
    OUT:
        NONE (writes to disk)
    """
    os.makedirs(os.path.dirname(INDEX_CACHE_PATH), exist_ok=True)
    # If the index is already persisted, re-persist the files, writing to temporary
    # files and copying over to overwrite existing files once done.
    try:
        # Write temporary files
        faiss.write_index(index, str(TEMP_INDEX_CACHE_PATH))
        with open(TEMP_METADATA_CACHE_PATH, "wb") as f:
            pickle.dump(doc_metadata, f)
    except Exception as e:
        print(f"Hit error {e} while writing overwriting index and metadata, skipping replacement")
        raise e
    os.replace(TEMP_INDEX_CACHE_PATH, INDEX_CACHE_PATH)
    os.replace(TEMP_METADATA_CACHE_PATH, METADATA_CACHE_PATH)
        

def ingest(config: dict[str, Any], embed_model: models.GoogleGenAIEmbed) -> tuple[faiss.IndexFlatIP, dict[str, Any]]:
    """
    Driver upon start up, loads vector index and metadata if it exists or 
    runs ingestion from scratch based on files from
    IN:
        config: Application config file, expected to exist in <root>/<src>/code_sample_rag/app_config.json
        embed_model: models.GoogleGenAIEmbed object, used to create embeddings
    OUT:
        List of tuple of form 
        (<FAISS_index>, <list_of_metadata_JSONs>)
        Where the metadata JSON has one entry per chunk, and is of the form
        {
            "start_page": <starting_page_num>,
            "end_page": <ending_page_num>,
            "chunk": <chunk_content>,
            "file_name": <file_name>
        }
    """
    try:
        index = faiss.read_index(str(INDEX_CACHE_PATH))
        with open(METADATA_CACHE_PATH, "rb") as f:
            all_metadata = pickle.load(f)
        return index, all_metadata
    except Exception as e:
        print(f"Hit the following error while reading index or metadata, re-ingesting instead:\n{e}")

    # Set up chunking object and embedding model
    chunk_config = config["chunking"]
    chunker = chunking.CharChunking(chunk_config)

    all_metadata: list[dict[str, Any]] = []
    all_text_chunks = []
    for file_path in os.listdir(DATA_PATH):
        if os.path.splitext(file_path)[-1] == ".pdf":
            full_file_path = DATA_PATH / file_path
            parser = parsing.PDFParser(full_file_path)
            
            parsed_text, page_end_indicies = parser.parse()
            chunks_data = chunker.chunk(parsed_text, page_end_indicies)

            # NOTE: The embeddings, text chunks and all metadata have
            # to be kept in the same order in order to properly display
            # chunks during retrieval
            text_chunks = []
            file_name = os.path.basename(file_path)
            for text_chunk, start_page_num, end_page_num in chunks_data:
                text_chunks.append(text_chunk)
                current_metadata = {
                    "start_page": start_page_num,
                    "end_page": end_page_num,
                    "chunk": text_chunk,
                    "file_name": file_name
                }
                all_metadata.append(current_metadata)
            all_text_chunks.extend(text_chunks)

    print(f"Will embed {len(all_text_chunks)} chunks for the whole environment")
    embeddings = embed_model.embed_text(all_text_chunks, task_type_str=models.TASK_TYPE_001_RET_DOC)
    index = create_vec_idx(embeddings)

    persist_data(index, all_metadata)
    return index, all_metadata
