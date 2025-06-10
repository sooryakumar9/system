from pydantic import BaseModel
from typing import List
import wikipedia
from langchain.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_cohere import ChatCohere

class InstitutionInfo(BaseModel):
    founder: str
    founded_year: str
    branches: List[str]
    num_employees: str
    summary: str

parser = PydanticOutputParser(pydantic_object=InstitutionInfo)

prompt = PromptTemplate(
    template="""
You are a helpful assistant extracting facts from Wikipedia articles.
Given this article text, extract the following about the institution:

- Founder
- Founded year
- List of current branches
- Approximate number of employees
- A short 4-line summary

Wikipedia Article:
{text}

Use this format:
{format_instructions}
""",
    input_variables=["text"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

llm = ChatCohere(
    cohere_api_key="",  # ⬅️ Replace this with your key
    model="command",
    temperature=0.3
)
chain: RunnableSequence = prompt | llm | parser

institution_name = input("Enter the name of the Institution: ")

try:
    wiki_text = wikipedia.page(institution_name).content
except wikipedia.exceptions.DisambiguationError as e:
    print("Multiple pages found. Suggestions:", e.options)
    exit()
except wikipedia.exceptions.PageError:
    print("Page not found on Wikipedia.")
    exit()

try:
    result: InstitutionInfo = chain.invoke({"text": wiki_text})

    print("\nInstitution Details:\n")
    print(f"Founder: {result.founder}")
    print(f"Founded Year: {result.founded_year}")
    print(f"Branches: {', '.join(result.branches)}")
    print(f"Employees: {result.num_employees}")
    print(f"\nSummary:\n{result.summary}")

except Exception as e:
    print("Error during processing:", e)