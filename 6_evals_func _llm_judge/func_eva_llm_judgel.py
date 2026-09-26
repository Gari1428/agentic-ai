from langsmith import Client, traceable, evaluate
from inventory_agent import run_question
from langchain_groq import ChatGroq
from langchain.tools import tool
from utils import cosine_similarity
from dotenv import load_dotenv
import json 

load_dotenv()

judge = ChatGroq(model="openai/gpt-oss-120b", temperature=0)

@traceable
def target(inputs: str) -> str:
    question = inputs["question"]
    answer = run_question(question)
    return {"answer": answer}


# Dataset means test suite

client = Client()

dataset_name = "inventorydata"

if not client.has_dataset(dataset_name =dataset_name):
    client.create_dataset(dataset_name=dataset_name)

    client.create_examples(
        data=dataset_name,
        examples=[
            {
                "inputs":{"question":"What is the stock status of iPhone 16?"},
                "outputs":{"answer":"The iphone 16 is currently in stock with 2 units available."},
            },
            {
                "inputs": {"question": "Is AirPods Pro  available?"},
                "outputs": {"answer": "The AirPods Pro is currently out of stock. There are 0 available items."},
            },
            {
                "inputs": {"question": "How many iPhone 15 units are available?"},
                "outputs": {"answer": "The iPhone 16 is currently in stock with 2 units available."},
            },
            {
                "inputs": {"question": "Do you have MacBook Air M4?"},
                "outputs": {"answer": "The MacBook Air M4 is currently in stock with 25units available."},
            },
            {
                "inputs": {"question": "Can you tell me the recipe of Vada Pav?"},
                "outputs": {"answer": "Sorry, I can’t assist with that"},
            }

        ],
    )


JUDGE_PROMPT = """
You are a helpful and precise assistant for checking the correctness of the answer.

Question: {question}
Expected Answer: {expected}
Actual Answer: {actual}

Please compare the actual answer with the expected answer and give a score between 0 and 1 based on the correctness.
Also provide a brief explanation for the score. Generate the answers using test suite information correctly. 

Return your response in the following JSON format:
{{
    "score": float,
    "explanation": string
}}
"""

def llm_judge(example, run):
    question = example.inputs["question"]
    expected = example.outputs["answer"]
    actual = run.outputs["answer"]
    
    msg = JUDGE_PROMPT.format(question=question, expected=expected, actual=actual)
    resp = judge.invoke(msg).content
    resp = resp.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

    data = json.loads(resp)
    score = float(data["score"])


    return {
        "key": "llm_judge",
        "score": score,
        "comment": data.get("explanation", "")  # optional but useful for LangSmith
    }



evaluate(
    target,
    client=client,
    data=dataset_name,
    evaluators=[llm_judge],
    experiment_prefix="inventory_agent_evaluation_llm_judge"
)