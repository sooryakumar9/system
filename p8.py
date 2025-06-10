import os
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_cohere import ChatCohere

os.environ["COHERE_API_KEY"] = ""

file_path = '/Users/sooryakumar/PYTHON/text1.txt'
with open(file_path, 'r') as file:
    document_text = file.read()

llm = ChatCohere()

prompt = PromptTemplate(
    input_variables=["document"],
    template="""
        You are a helpful assistant.
        Given the following document, summarize it in bullet points:
        ---
        {document}
        ---
        Summary:
    """
)

chain = prompt | llm | StrOutputParser()

response = chain.invoke({"document": document_text})

print(response)