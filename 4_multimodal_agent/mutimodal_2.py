from dotenv import load_dotenv
from pathlib import Path
import base64
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
import yfinance as yf

load_dotenv()

llm = ChatGroq(model = "qwen/qwen3.8-27b", temperature=0)


@tool
def get_item_code(item_name:str)->str:
    """
    Return a unique item code for a given item name. Expected item names are: teddy bear, toy car, frock, puzzle, action figure.
    """
    if item_name == "teddy bear":
        return "ITM001"
    elif item_name == "toy car":
        return "ITM002"
    elif item_name == "frock":
        return "ITM003"     

def encode_image_to_data_url(path: Path) -> str:
    """Convert a local image file to a data URL usuable in image_url inputs."""
    with open(path, "rb") as image_file:
        b64=base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"


base_dir = Path.cwd()
img1_url = encode_image_to_data_url(base_dir / "images" / "image1.jpg")
img2_url = encode_image_to_data_url(base_dir / "images" / "image2.jpg")
img3_url = encode_image_to_data_url(base_dir / "images" / "image3.jpg")





SYSTEM_PROMPT = '''For each image, generate a JSON record that looks like this:
    {
        "item_name": "teddy bear",
        "item_code": "ITM001",
        "color": "blue",
        "age_category": "kids"
    }
    Output must be a JSON string that Python can parse it directly.
    Do not put any pre-amble instructions or event 'json' in front of the response string.
    item_name should be one of the following: teddy bear, toy car, frock, puzzle, action figure
    age_category should be one of the following: kids
    '''

human_content = [
    {"type": "text", "text": "Analyze each image and return a JSON array of records as instructed."},
    {"type": "image_url", "image_url": {"url": img1_url}},
    {"type": "image_url", "image_url": {"url": img2_url}},
    {"type": "image_url", "image_url": {"url": img3_url}},
]

agent = create_agent(
    tools = [get_item_code],
    model =llm,
    system_prompt=SYSTEM_PROMPT
)

result = agent.invoke(
    {
        "messages":[
        {
            "role":"user",
            "content":human_content
        }
        ]
    }
)

print(result["messages"][-1].content)
