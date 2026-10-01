from code_sample_rag import parsing, chunking, ingestion, retrieval, setup

if __name__ == "__main__":
    resources = setup.RAGResources()
    retrieval.retrieve_content(
        query="Which courses are offered in East Asian Studies department?",
        embed_model=resources.em,
        index=resources.index,
        doc_metadata=resources.doc_metadata,
        top_k=resources.config["retrieval"]["num_results"]
    )