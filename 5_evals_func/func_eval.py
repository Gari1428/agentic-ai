from langsmith import Client, traceable, evaluate
from inventory_agent import run_question
from langchain.tools import tool
from utils import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

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
                "outputs": {"answer": "The iPhone 16 is currently in stock with 7 units available."},
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


def semantic_match(example,run):
    expected=example.outputs["answer"]
    actual=run.outputs["answer"]
    sim = cosine_similarity(expected,actual)
    return {
        "key":"semantic_match",
        "score":float(sim)
    }



evaluate(
    target,
    client=client,
    data=dataset_name,
    evaluators=[semantic_match],
    experiment_prefix="inventory_agent_evaluation_openai/gpt-oss-120b"
)