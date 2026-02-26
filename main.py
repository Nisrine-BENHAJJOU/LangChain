from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq
from tavily import TavilyClient

tavily = TavilyClient()

load_dotenv()

@tool
def search(query: str) -> str:
    """Tool that searches over the internet"""
    print(f"Searching for: {query}")
    return tavily.search(query=query)

llm = ChatGroq(model="qwen/qwen3-32b")
tools = [search]
agent = create_agent(model=llm, tools=tools)

def main():
    print("Hello from langchain-course!")
    result = agent.invoke({
        "messages": [HumanMessage(content="Search for 3 job offers for an AI engineer using langchain in Morocco on linkedIn and list their details")]
    })
    print(result)

if __name__ == "__main__":
    main()