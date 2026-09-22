from dotenv import load_dotenv
import os

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_groq import ChatGroq
import yfinance as yf

load_dotenv()

llm = ChatGroq(model = "openai/gpt-oss-120b", temperature=0)

@tool
def get_stock_price(ticker:str)->str:
    """
    Get the latest stock price for a given ticker symbol using yfinance.
    """
    stock = yf.Ticker(ticker)
    price = stock.info.get("currentPrice")
    return f"The latest stock price for {ticker} is ${price}."

@tool
def get_market_valuation_of_private_company(company_name:str)->str:
    """
    Return the market valuation of a private company in billion USD. This is a placeholder function
    """
    # In a real implementation, you would use an API or database to fetch this data.
    # For demonstration purposes, we will return a fixed value.


    company_valuations = {
        "SpaceX": 137.0,  # Example valuation in billion USD
        "Stripe": 95.0,   # Example valuation in billion USD
        "Airbnb": 100.0,  # Example valuation in billion USD

}

    return company_valuations.get(company_name,0.0)

# Create agent
agent = create_agent(
    tools = [get_stock_price, get_market_valuation_of_private_company],
    model =llm,
    system_prompt=("You are a Finance agent")
)

result = agent.invoke(
    {
        "messages":[
        {
            "role":"user",
            "content":(
                "What is the market valuation of Stripe?"
            ),
        }
        ]
    }
)

print(result["messages"][-1].content)