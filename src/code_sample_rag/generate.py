# Imports
# Python
import json
from typing import Any, Optional

# Codebase
from code_sample_rag import retrieval, models

# Given retrieved documents and prompt from config,
# call LLM with prompt

# TODO: Add configurable guardrails layer based on keywords
CONTEXT_HEADER_FMT = "File: {file_name} from pages {start_page} to {end_page}"

### FUNCS ###
def construct_prompt(config: dict[str, Any],
                     query: str,
                     retrieved_results: list[dict[str, Any]]) -> str:
    """
    Given prompt engineering information, query and retrieved results,
    output a RAG prompt for the LLM
    IN:
        config: Application config file, expected to exist in <root>/<src>/code_sample_rag/app_config.json
        query: Query to be answered
        retrieved_results: An ordered list of retrieved chunk metadata JSONs, each of the form
                           { 
                               "start_page": <starting_page_num>,
                               "end_page": <ending_page_num>,
                               "chunk": <chunk_content>,
                               "file_name": <file_name>
                               "cos_sim": <cosine_similarity>
                           }
    OUT:
    """
    prompt_components = []
    # The prompt is expected to look as follows:
    # <context_instructions>
    # <section_separator>
    # <context_format> + <retrieved_result_1>
    # <section_separator>
    # ...
    # <section_separator>
    # <question_format> + <query>
    sec_sep = config["generate"]["section_separator"]
    context_format = config["generate"]["context_format"]

    prompt_components.append(config["generate"]["answering_instructions"])
    for result in retrieved_results:
        # Set up header
        header = CONTEXT_HEADER_FMT.format(
            file_name=result["file_name"],
            start_page=result["start_page"],
            end_page=result["end_page"]
        )
        context_text = header + "\n" + result["chunk"]
        context_section = context_format.format(context=context_text)
        prompt_components.append(context_section)
    question_section = config["generate"]["question_format"].format(question=query)
    prompt_components.append(question_section)

    return sec_sep.join(prompt_components)


def answer(query: str,
           config: dict[str, Any],
           retrieved_results: list[dict[str, Any]],
           llm: models.GoogleGenAILLM,
           prev_inter_id: Optional[str]=None) -> str:
    """
    Answer the query using retrieved results
    IN:
        query: Query to be answered
        config: Application config file, expected to exist in <root>/<src>/code_sample_rag/app_config.json
        retrieved_results: An ordered list of retrieved chunk metadata JSONs, each of the form
        llm: models.GoogleGenAILLM object, used to generate response
        prev_inter_id: Previous conversation ID for multi-turn
    OUT:
        tuple of the form (<llm_response>, <interaction_id>)
    """
    system_instruction = config["generate"]["system_instruction"]
    generation_config = config["models"]["LLM"]["generation_config"]
    full_prompt = construct_prompt(config, query, retrieved_results)
    
    answer_text, inter_id = \
        llm.generate_text(text=full_prompt,
                          system_instruction=system_instruction,
                          generation_config=generation_config,
                          prev_inter_id=prev_inter_id)

    return answer_text, inter_id
