from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
import yfinance as yf

load_dotenv()

llm = ChatGroq(model = "openai/gpt-oss-120b", temperature=0)

agent = create_agent(
    tools = [],
    model =llm,
    system_prompt="""
    You are an image agent
"""
)

result = agent.invoke(
    {
        "messages":[
        {
            "role":"user",
            "content": "Which bird is in this image? print the name of bird along with some information about it: https://pixabay.com/illustrations/ai-generated-peacock-bird-animal-8909981/ "
        }
        ]
    }
)

print(result["messages"][-1].content)