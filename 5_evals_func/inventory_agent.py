from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain.tools import tool



load_dotenv()

llm = ChatGroq(model = "openai/gpt-oss-120b", temperature=0)

@tool
def inventory_tool(product_name:str)->str:
    """Check inventory availability for a given product name."""
    print(f"TOOL CALLED for: {product_name}")
    inventory = {
        "iPhone 17": "In Stock: Available Items = 2",
        "Airpods Pro 3": "Out of Stock: Available Items = 0",
        "MacBook Air M4": "In Stock: Available Items = 5",
    }
    return inventory.get(product_name, "Product not found in inventory.")

#Inventory agent
agent = create_agent(
    tools = [inventory_tool],
    model = llm,
    system_prompt="""
    You are an inventory assistant. 

    - If question is out of scope which is not related to inventory then just say "Sorry, I can't assisst you with that." No need to say anything else.

    When a user asks about a product, use the inventory_tool to fetch the inventory data.

    - Always call the inventory_tool with full product name.
    - inventory_tool will return a dictionary which you need to parse to extract stock status, inventory items etc
    -  Respond with clear and concise information including:
        1. The stock status (eg. In Stock, Out of Stock)
        2. The number of items available in stock (if applicable)
        - If the product is not found, say: "The product is not available in our inventory."

    Never guess or hallucinate information. Do not respond unless the inventory_tool is called.
    Keep your response short and informative.
    """ 
)   

def run_question(question):
    result = agent.invoke(
        {
            "messages":[
            {
                "role":"user",
                "content": question
            }
            ]
        }
    )

    return result["messages"][-1].content


if __name__ == "__main__":
    question = " What's the inventory status of MacBook Air M4?"
    print(run_question(question))