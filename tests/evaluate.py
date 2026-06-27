import json
from pathlib import Path
import requests
import os
from dotenv import load_dotenv
from ollama import GenerateResponse, Client

load_dotenv()
api_host = os.getenv("API_HOST")
api_port = os.getenv("API_PORT")
ollama_model = os.getenv("OLLAMA_MODEL")
ollama_base_url = os.getenv("OLLAMA_BASE_URL") 

ollama_client = Client(host= ollama_base_url)

def create_test_dict(filename : str) -> list[dict]:
    test_case_list = []
    with open(filename, "r") as file:
        data = json.load(file)
    for entry in data:
        test_case = {
            "question" : entry["question"],
            "expected_answer" : entry["expected_answer"],
            "generated_answer" : "",
            "expected_sources" : entry["expected_sources"],
            "retrieved_sources" : []
        }
        test_case_list.append(test_case)
    return test_case_list


def process_questions(test_case_list: list[dict], collection_name: str = "default") -> list[dict]:
    base_url = f"http://{api_host}:{api_port}"
    for test_case in test_case_list:
        query = test_case["question"]
        response = requests.post(
            f"{base_url}/query/",
            json={"body": query},
            params={"collection_name": collection_name}
        )
        data = response.json()
        test_case["generated_answer"] = data["answer"]
        test_case["retrieved_sources"] = list(dict.fromkeys(
            [source["filepath"] for source in data["sources"]]
        ))
    return test_case_list


def score_retrieval(test_case: dict) -> float:
    expected_sources = set(test_case["expected_sources"])
    retrieved_sources = set(test_case["retrieved_sources"])
    expected_count = len(expected_sources)
    intersect_count = len(expected_sources & retrieved_sources)
    return intersect_count / expected_count


def score_answer(test_case: dict) -> float:
    expected_answer = test_case["expected_answer"]
    generated_answer = test_case["generated_answer"]

    prompt = f"""You are an evaluation assistant. Compare the following two answers and rate their similarity on a scale from 0.0 to 1.0, where:
1.0 = the answers are semantically identical or convey the same information
0.5 = the answers partially overlap but one is missing key information
0.0 = the answers are completely different or contradictory

Respond with ONLY a single float number between 0.0 and 1.0. No explanation, no other text.

Expected Answer:
{expected_answer}

Generated Answer:
{generated_answer}

Score:"""
    
    response = ollama_client.generate(
        model= ollama_model,
        prompt= prompt
    )
    try:
        return float(response.response.strip())
    except ValueError:
        return 0.0


def calculate_metrics (test_case_list: list[dict]) -> dict:
    summary_dict={
        "avg_retrieval_score" : 0,
        "avg_answer_score" : 0,
        "question_summary" : []
    }
    tot_retrieval_score = 0
    tot_answer_score = 0
    test_num = len(test_case_list)
    for test_case in test_case_list:
        retrieval_score = score_retrieval(test_case)
        answer_score = score_answer(test_case)
        test_case["retrieval_score"] = retrieval_score
        test_case["answer_score"] = answer_score
        tot_retrieval_score += retrieval_score
        tot_answer_score += answer_score
        summary_dict["question_summary"].append(test_case)
    summary_dict["avg_retrieval_score"] = tot_retrieval_score / test_num
    summary_dict["avg_answer_score"] = tot_answer_score / test_num
    return summary_dict


test_file = Path(__file__).resolve().parent / "test_questions.json"
test_dict = create_test_dict(test_file)
full_test_case_dict = process_questions(test_dict)
metrics = calculate_metrics(full_test_case_dict)
print(f"Average Retrieval Score: {metrics['avg_retrieval_score']:.2f}")
print(f"Average Answer Score: {metrics['avg_answer_score']:.2f}")