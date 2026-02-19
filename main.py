from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama

load_dotenv() 

def main():
    print("Hello from langchain-course!")
    
    # 1. This is the raw data about Isaac Newton that we want to propagate to the LLM
    information = """
    Sir Isaac Newton was an English polymath who was a mathematician, physicist, astronomer, alchemist, theologian, author and inventor. He was a key figure in the Scientific Revolution and the Enlightenment that followed. His book Philosophiæ Naturalis Principia Mathematica (Mathematical Principles of Natural Philosophy), first published in 1687, achieved the first great unification in physics and established classical mechanics. Newton also made seminal contributions to optics, and shares credit with the German mathematician Gottfried Wilhelm Leibniz for formulating infinitesimal calculus, although he developed calculus years before Leibniz. Newton contributed to and refined the scientific method, and his work is considered the most influential in bringing forth modern science.
    """

    # 2. Define the template string with a {placeholder}
    # This placeholder is not permanent; it will be substituted at runtime
    summary_template = """
    Given the information {information} about a person, I want you to create:
    1. A short summary
    2. two interesting facts about them
    """

    # 3. Initialize the PromptTemplate (a "First Class Citizen" in LangChain)
    # We use this instead of f-strings because:
    # - It enforces strict formatting (safer against prompt injection)
    # - It provides clarity & reusability across different chains
    # - It enables logging and tracing (integrated with LangSmith)
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], # Must match the curly brackets in the string
        template=summary_template
    )

    #4. Initialize the LLM (ChatOllama in this case)
    llm = ChatOllama(model="gemma3:4b", temperature=0)

    #5. Create a runnable chain by connecting the PromptTemplate to the LLM
    chain = summary_prompt_template | llm

    #6. Invoke the chain with the raw data as input, and print the response
    response = chain.invoke(input={"information": information})
    print(response.content)

if __name__ == "__main__":
    main()