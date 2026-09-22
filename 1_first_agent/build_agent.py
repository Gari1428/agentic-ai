from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain_community.utilities import SerpAPIWrapper
from langchain.tools import tool
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(model = "openai/gpt-oss-120b", temperature=0)

# Fetch SerpAPI key
serpapi_key = os.getenv("SERPAPI_KEY")


serp = SerpAPIWrapper(
    serpapi_api_key=serpapi_key,
    params={
        "tbm": "nws",      # Search in Google News
        "tbs": "qdr:d",    # Past day (24 hours)
    },
)

@tool
def search_news(query:str)->str:
    """
    Search last 24 hrs newsGoogle News using SerpAPI.
    Return news results with URLs.
    """
    return serp.run(query)

# Create agent
agent = create_agent(
    tools = [search_news],
    model =llm,
    system_prompt=("You are a news reporter who reports news using simple langauage, in 3 lines along with the date of news")
)

result = agent.invoke(
    {
        "messages":[
        {
            "role":"user",
            "content":(
                "Tell me the latest breaking news in India, along with the web url of news "
            ),
        }
        ]
    }
)

print(result["messages"][-1].content)