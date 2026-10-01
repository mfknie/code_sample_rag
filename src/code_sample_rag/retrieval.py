# Imports
# Python
from copy import deepcopy
from typing import Any, Optional

# Codebase
from code_sample_rag import models

# 3rd-party
import faiss
import numpy as np

# Given a query, embed the query's text using the canonical embedding model 
# set up in embedding module
def embed_query(query: str, embed_model: models.GoogleGenAIEmbed):
    """
    IN:
        query: Query to be embedded
        embed_model: GoogleGenAIEmbed object used to generate the embedding
    OUT:
    NOTES:
        Only one embedding model can be specified in the application config, so it is impossible
        for there to be an embedding mismmatch between the vector store and the embedded query
    """
    embeddings = embed_model.embed_text(texts=[query], task_type_str=models.TASK_TYPE_001_RET_QUERY)
    # Normalize embedings
    embeddings_norm = np.linalg.norm(embeddings, axis=1, keepdims=1)
    return embeddings/embeddings_norm

# Search the query embedding with cosine similarity against values in vector store
# Pick the top K (configurable) results, with text and link back to document
def retrieve_context(query: str,
                     embed_model: models.GoogleGenAIEmbed,
                     index: faiss.IndexFlatIP,
                     doc_metadata: list[dict[str, Any]],
                     top_k: int=5) -> list[dict[str, Any]]:
    """
    IN:
        query: Query to be embedded 
        embed_model: GoogleGenAIEmbed object used to generate the embedding
        index: faiss.IndexFlatIP over stored vectors to use for vector search
        doc_metadata: metadata for chunks associated with stored vectors (in same order)
        top_k: Top K results to be retrieved and used for RAG
    OUT:
        An ordered list of metadata JSONs, each of the form
        { 
            "start_page": <starting_page_num>,
            "end_page": <ending_page_num>,
            "chunk": <chunk_content>,
            "file_name": <file_name>
            "cos_sim": <cosine_similarity>
        }
    NOTES:
        TODO: Support other types of similarity search
        TODO: Create dataclass of metadata
    """
    retrieved_results = []

    query_embedding = embed_query(query, embed_model)
    # Note that similarities and indicies only contain one row each,
    # since there was only one query
    # NOTE: FAISS documentation uses the variable D (distances) for
    # IndexFlatIP, even though if the vectors are normalized, the
    # cosine similarity is maximized for during the search process
    # and returned
    similarities, indicies = index.search(query_embedding, top_k)

    # Only retain first row, since there is only one query
    # TODO: Add minimum similarity score for inclusion in results
    for sim, ind in zip(similarities[0], indicies[0]):
        current_metadata = deepcopy(doc_metadata[ind])
        current_metadata["cos_sim"] = float(sim)
        retrieved_results.append(current_metadata)

    return retrieved_results
