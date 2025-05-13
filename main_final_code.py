from utils_retrieval import query_pinecone, query_opensearch, BGEReranker
from prompt_library import build_prompt
from model_hadler import ModelHandler
from utils import append_to_jsonl
from jsonschema import validate
from itertools import islice
from tqdm import tqdm
import pandas as pd
import jsonschema
import argparse
import json
import os
import re
import time
import logging


# Configure logging
logging.basicConfig(filename="timing_log_part2.log", level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
total_start = time.time()  # Time


# ========================================
# Section: Call LLM
# ========================================


def call_llm(messages, handler, **kwargs):
    response = handler.generate_answer(messages, **kwargs)
    return response.replace('"', "").strip()


# ========================================
# Section: Classify Question Rule
# ========================================

comparison_keywords = [
    "differ",
    "difference",
    "different",
    "compare",
    "comparison",
    "between",
    "compare to",
    "compare between",
    "differ to",
    "difference between",
]
wh_words = ["what", "when", "where", "which", "who", "whom", "whose", "why", "how"]


def classify_question_rule_based(question):
    q = str(question).lower().strip()
    if any(keyword in q for keyword in comparison_keywords):
        return "multidoc"
    if len(re.findall(r"\b(" + "|".join(wh_words) + r")\b", q)) >= 2:
        return "multidoc"
    if q.startswith("which"):
        return "multidoc"
    return "single"


# ========================================
# Section: Validation Answers
# ========================================


def check_answers(root_answers):

    json_schema_answer = """
    { 
    "$schema": "http://json-schema.org/draft-07/schema#", 

    "title": "Answer file schema", 
    "type": "object", 
    "properties": { 
        "id": { 
        "type": "integer", 
        "description": "Question ID" 
        }, 
        "question": { 
        "type": "string", 
        "description": "The question" 
        }, 
        "passages": { 
        "type": "array", 
        "description": "Passages used and related FineWeb doc IDs, ordered by decreasing importance", 
        "items": { 
            "type": "object", 
            "properties": {
            "passage": { 
                "type": "string", 
                "description": "Passage text" 
            }, 
            "doc_IDs": {
                "type": "array", 
                "description": "Passage related FineWeb doc IDs, ordered by decreasing importance", 
                "items": { 
                "type": "string", 
                "description": "FineWeb doc ID, e.g., <urn:uuid:d69cbebc-133a-4ebe-9378-68235ec9f091>"
                } 
            } 
            },
            "required": ["passage", "doc_IDs"]
        }
        }, 
        "final_prompt": {
        "type": "string",
        "description": "Final prompt, as submitted to Falcon LLM"
        },
        "answer": {
        "type": "string",
        "description": "Your answer"
        }
    },
    "required": ["id", "question", "passages", "final_prompt", "answer"]
    }
    """

    # Load the file to make sure it is ok
    loaded_answers = pd.read_json(root_answers, lines=True)

    # Load the JSON schema
    schema = json.loads(json_schema_answer)

    # Validate each Answer JSON object against the schema
    for answer in loaded_answers.to_dict(orient="records"):
        try:
            validate(instance=answer, schema=schema)
            # print(f"Answer {answer['id']} is valid.")
        except jsonschema.exceptions.ValidationError as e:
            print(f"Answer {answer['id']} is invalid: {e.message}")


def handle_multidoc(question, backend_LLM, topNumber, reranker):
    new_questions = []
    all_merged_contexts = []

    messages = build_prompt(prompt_key="convert_multidoc_questions", question=question)
    convert_questions = call_llm(messages, backend_LLM)
    simpler_questions = re.findall(r"Question\d+:\s*(.*?\?)", convert_questions)

    for individual_question in simpler_questions:
        new_questions.append(individual_question)
        results_sparse = query_opensearch(individual_question, top_k=200)
        text_context_sparse_init = [match["_source"]["text"] for match in results_sparse["hits"]["hits"]]
        id_list_sparse = [re.findall(r"<[^>]*>", match["_id"])[0] for match in results_sparse["hits"]["hits"]]

        results_dense = query_pinecone(individual_question, top_k=200)
        text_context_dense_init = [match["metadata"]["text"] for match in results_dense["matches"]]
        id_list_dense = [match["metadata"]["doc_id"] for match in results_dense["matches"]]

        # =============================
        # Section: Hybrid Search
        # =============================

        hybrid_text_context_init = text_context_sparse_init[0:100] + text_context_dense_init[0:100]
        hybrid_id_list = id_list_sparse[0:100] + id_list_dense[0:100]
        # =============================
        # Section: Filter duplicated documents*
        # =============================
        # hybrid_unique_documents = list(set(hybrid_text_context_init))
        documents = set()
        hybrid_unique_documents = []
        hybrid_unique_list_id = []
        for hybrid_text, hybrid_id in zip(hybrid_text_context_init, hybrid_id_list):
            if hybrid_text not in documents:
                documents.add(hybrid_text)
                hybrid_unique_documents.append(hybrid_text)
                hybrid_unique_list_id.append(hybrid_id)

        # =============================
        # Section: Reranker
        # =============================
        start = time.time()
        ranked_list_docs, score_hybrid = reranker.rerank(question, hybrid_unique_documents, 100)
        ranked_text_context = [hybrid_unique_documents[index] for index in ranked_list_docs]
        ranked_id_context = [hybrid_unique_list_id[index] for index in ranked_list_docs]
        end = time.time()
        merged_contexts = []
        for text, id_, score in zip(ranked_text_context, ranked_id_context, score_hybrid):
            merged_contexts.append({"text": text, "id": id_, "score": score})
        all_merged_contexts.append(merged_contexts)

        # merged_contexts.append({
        #     "question": individual_question,
        #     "texts": ranked_text_context,
        #     "ids": ranked_id_context,
        #     "scores" : ranked_scores
        # })

    # Merge texts and ids
    # Duplicate document are ranked first
    if not all_merged_contexts:
        print("-------------------------Error------------------------------------------------")
        return [], simpler_questions
    all_items = []
    for items1, items2 in islice(zip(all_merged_contexts[0], all_merged_contexts[1]), topNumber):
        all_items.append(items1)
        all_items.append(items2)

    documents = set()
    context = []
    for item in all_items:
        if item["text"] not in documents:
            documents.add(item["text"])
            context.append(item)

    return context, simpler_questions


def handle_singledoc(question, backend_LLM, reranker):

    messages = build_prompt(prompt_key="TEST#sparse", query=question)
    question_re_write = call_llm(messages, backend_LLM)
    question_re_write_sparse = question_re_write.split(":**")[-1].strip()

    messages = build_prompt(prompt_key="TEST#dense", query=question)
    question_re_write_dense = call_llm(messages, backend_LLM).split(":**")[-1].strip()

    # logging.info(f"Rewrite queries took {end - start:.4f} seconds")
    # =============================
    # Section: Retrieval
    # =============================
    start = time.time()
    results_sparse = query_opensearch(question_re_write_sparse, top_k=200)
    text_context_sparse_init = [match["_source"]["text"] for match in results_sparse["hits"]["hits"]]
    id_list_sparse = [re.findall(r"<[^>]*>", match["_id"])[0] for match in results_sparse["hits"]["hits"]]

    results_dense = query_pinecone(question_re_write_dense, top_k=200)
    text_context_dense_init = [match["metadata"]["text"] for match in results_dense["matches"]]
    id_list_dense = [match["metadata"]["doc_id"] for match in results_dense["matches"]]
    end = time.time()
    logging.info(f"Retrival Section took {end - start:.4f} seconds")
    # =============================
    # Section: Hybrid Search
    # =============================

    hybrid_text_context_init = text_context_sparse_init[0:100] + text_context_dense_init[0:100]
    hybrid_id_list = id_list_sparse[0:100] + id_list_dense[0:100]
    # =============================
    # Section: Filter duplicated documents*
    # =============================
    # hybrid_unique_documents = list(set(hybrid_text_context_init))
    documents = set()
    hybrid_unique_documents = []
    hybrid_unique_list_id = []
    for hybrid_text, hybrid_id in zip(hybrid_text_context_init, hybrid_id_list):
        if hybrid_text not in documents:
            documents.add(hybrid_text)
            hybrid_unique_documents.append(hybrid_text)
            hybrid_unique_list_id.append(hybrid_id)

    # =============================
    # Section: Reranker
    # =============================
    start = time.time()
    ranked_list_docs, score_hybrid = reranker.rerank(question, hybrid_unique_documents, 100)
    ranked_text_context = [hybrid_unique_documents[index] for index in ranked_list_docs]
    ranked_id_context = [hybrid_unique_list_id[index] for index in ranked_list_docs]
    end = time.time()
    logging.info(f"Reranker Section took {end - start:.4f} seconds")
    merged_contexts = []
    for text, id_, score in zip(ranked_text_context, ranked_id_context, score_hybrid):
        merged_contexts.append({"text": text, "id": id_, "score": score})

    return merged_contexts, [question]


def process_jsonl(file_path, backend_LLM, back_rerank):

    root_dir, ext = os.path.splitext(file_path)
    root_answers = f"answers_{root_dir}.jsonl"

    if os.path.exists(root_answers):
        os.remove(root_answers)
        print("Deleted answer file")

    topNumberSingle = 5
    topNumberMulti = 3

    if ext.lower() == ".jsonl":
        loaded_questions = pd.read_json(file_path, lines=True)
        print("Loaded JSONL file.")
    elif ext.lower() == ".csv":
        loaded_questions = pd.read_csv(file_path)
        print("Loaded CSV file.")

    for index, row in tqdm(loaded_questions.iterrows()):
        idQuestion = row["id"]
        print("_-_-_-_ Processing idQuestion: ", idQuestion)
        question = row["question"]
        logging.info(f"ID Question {idQuestion}")
        logging.info(f"Question {question}")
        start = time.time()

        # =============================
        # Section: Classification Prompt
        # =============================
        classification = classify_question_rule_based(question)
        logging.info(f"Classification {classification}")

        # =============================
        # Section: Pre Generation Answer
        # =============================
        passages_list = []

        if classification == "multidoc":
            results, subquestions = handle_multidoc(question, backend_LLM, topNumberMulti, back_rerank)

            context = "\n\n".join(f"Document{i+1}:\n{items['text'].strip()}" for i, items in enumerate(results))

            for item in results:
                passages_data = {"passage": item["text"], "doc_IDs": [item["id"]]}
                passages_list.append(passages_data)

        else:
            results_all, subquestions = handle_singledoc(question, backend_LLM, back_rerank)
            results = results_all[:topNumberSingle]

            context = "\n\n".join(f"Document{i+1}:\n{text['text'].strip()}" for i, text in enumerate(results))
            # context = f"Document1:\n{results[0]['text'].strip()}"

            for item in results:
                passages_data = {"passage": item["text"], "doc_IDs": [item["id"]]}
                passages_list.append(passages_data)

        # =============================
        # Section: Generation Answer
        # =============================
        start = time.time()
        is_natural = "?" in question
        if is_natural:
            prompt_generated_answer = build_prompt(prompt_key="is_natural", context=context, question=question)
        else:
            prompt_generated_answer = build_prompt(prompt_key="is_query_style", context=context, question=question)

        response = call_llm(messages=prompt_generated_answer, handler=backend_LLM, max_new_tokens=1024, do_sample=False)
        end = time.time()
        logging.info(f"Generation Section took {end - start:.4f} seconds")
        logging.info(f"Answer {response}")

        # =============================
        # Section: Save Data Jsonl
        # =============================

        data = {
            "id": idQuestion,
            "question": question,
            # "sub_question": subquestions,
            "answer": response,
            #"Classification": classification,
            "passages": passages_list,
            "final_prompt": str(prompt_generated_answer),
        }

        append_to_jsonl(root_answers, data)
        end = time.time()
    logging.info(f"Total Execution took {end - total_start:.4f} seconds")
    print(f"Total Execution took {end - total_start:.4f} seconds")
    return root_answers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process a JSONL file.")
    parser.add_argument("jsonl_file", help="Path to the input .jsonl file")
    parser.add_argument("--cuda", default="cuda:1", help="CUDA device to use, e.g., cuda:0 or cuda:1")
    args = parser.parse_args()

    handler_a171 = ModelHandler(model_name="tiiuae/falcon3-10b-instruct", backend_type="AI71")


    print("Starting Process...")
    initial_file = args.jsonl_file
    reranker_bge_v2_m3 = BGEReranker(cuda_device=args.cuda)

    answers = process_jsonl(initial_file, handler_a171, reranker_bge_v2_m3)
    check_answers(answers)
