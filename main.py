from typing import List
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

class Source(BaseModel):
    """Schema for a source used by the agent"""

    url:str = Field(description="The URl of the source")

class AgentResponse(BaseModel):
    """Schema for agent response with answer and sources"""

    answer: str = Field(description="The agent's answer to the query")
    sources: List[Source] = Field(default_factory=list, description="List of sources used by the agent to answer the query")

llm = ChatOllama(model="qwen3:8b")
tools = [TavilySearch()]
agent = create_agent(model=llm, tools=tools, response_format=ToolStrategy(AgentResponse))

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        "messages": [HumanMessage(content="Search for 3 job offers for an AI engineer using langchain in Morocco on linkedIn and list their details")]
    })
    print(result)

if __name__ == "__main__":
    main()