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
    You are an advanced reasoning assistant.
    List out the steps you would take to answer a question with numbers.
    If using any formulas, it should not be in Latex format, but in plain formulas
"""
)

result = agent.invoke(
    {
        "messages":[
        {
            "role":"user",
            "content": "What's the area of circle with radius 4?"
        }
        ]
    }
)

print(result["messages"][-1].content)