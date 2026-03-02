from dotenv import load_dotenv

load_dotenv()

from langchain.chat_models import init_chat_model
from langchain.tools import tool

# human message for user input, system message for instructions to the LLM, tool message containing tool result
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage

from langsmith import traceable

MAX_ITERATIONS = 10
MODEL = "qwen3:8b"

# ---- Tools ---- LangChain @tool decorator


@tool
def get_product_price(product: str) -> float:
    """Look up the price of a product in the catalog."""
    print(f" >> Executing tool: get_product_price with input: {product}")
    prices = {"laptop": 999.99, "smartphone": 499.99, "headphones": 199.99}
    return prices.get(product.lower(), 0.0)


@tool
def apply_discount(price: float, discount_tier: str) -> float:
    """Apply discount tier to a price and return the final price.
    Discount tiers: 'bronze' = 10%, 'silver' = 20%, 'gold' = 30%
    """
    print(
        f" >> Executing tool: apply_discount with input: price={price}, discount_tier={discount_tier}"
    )
    discounts = {"bronze": 0.10, "silver": 0.20, "gold": 0.30}
    discount = discounts.get(discount_tier.lower(), 0.0)
    return round(price * (1 - discount))


# ---- Agent Loop ----

@traceable(name="LangChain Agent Loop")
def run_agent(question: str):
    tools = [get_product_price, apply_discount]
    tools_dict = {tool.name: tool for tool in tools}

    llm = init_chat_model(f"ollama:{MODEL}", temperature=0)
    llm_with_tools = llm.bind_tools(tools) # it'll only work for LLM supporting tool calling (like Ollama Qwen3)

    print(f"User question: {question}")
    print("=" * 80)

    messages = [
        SystemMessage(
            content=(
                "You are a helpful shopping assistant. "
                "You have access to a product catalog tool "
                "and a discount tool.\n\n"    
                "STRICT RULES - you must follow these exactly:\n"
                "1. NEVER guess or assume any product price. " 
                "You MUST call get_product_price first to get the real price.\n"
                "2. Only call apply_discount AFTER you have received "
                "a price from get_product_price. Pass the exact price. "
                "returned by get_product_price - do NOT pass a made-up number.\n"
                "3. Never calculate discounts yourself using math. "
                "Always use the apply_discount tool to get the correct discounted price.\n"
                "4. If the user does not specify a discount tier, "
                "ask them which tier to use - do NOT assume one."
            )
        ),
        HumanMessage(content=question),
    ]

    for iteration in range(1, MAX_ITERATIONS + 1):
        print(f"\n--- Iteration {iteration} ---")
        ai_message = llm_with_tools.invoke(messages)

        tool_calls = ai_message.tool_calls

        # if no tool calls this is the final answer
        if not tool_calls:
            print("Final answer from agent:")
            return ai_message.content
        
        # process only the first tool call - force one tool per iteration
        tool_call = tool_calls[0]
        tool_name = tool_call.get("name")
        tool_args = tool_call.get("args", {}) 
        tool_call_id = tool_call.get("id")

        print(f"  [Tool selected] {tool_name} with args {tool_args}")

        tool_to_use = tools_dict.get(tool_name)
        if tool_to_use is None:
            raise ValueError(f"Tool {tool_name} not found in tools_dict")
        
        observation = tool_to_use.invoke(tool_args)

        print(f"  [Observation] {observation}")

        messages.append(ai_message) # add the AI message with tool call to the context
        messages.append(
            ToolMessage(content=str(observation), tool_call_id=tool_call_id)
        ) # add the tool observation to the context
    
    print("Error: Max iterations reached without final answer.")
    return None


if __name__ == "__main__":
    print("Hello LangChain Agent (.bind_tools)!")
    print()
    result = run_agent("What is the price of a laptop with a silver discount?")
    print(result)
