from langchain import hub
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain_community.llms import Ollama
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_community.agent_toolkits import create_sql_agent
from langchain.agents import (AgentExecutor, Tool, ZeroShotAgent,
                              initialize_agent, load_tools)
from langchain.agents import Tool,create_react_agent
#from langchain.utilities import PythonREPL
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_openai_functions_agent
from langchain_core.prompts import ChatPromptTemplate,PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
import requests
from bs4 import BeautifulSoup

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import UnstructuredPDFLoader,PyPDFLoader
from graphviz import Source
#from strip_tags import strip_tags



class Output(BaseModel):
    content: str = Field(description="The content where the output is stored.")


model = OllamaLLM(model="llama3.1")
#model = ChatOllama(model='llama3.1')


base_prompt = hub.pull("langchain-ai/react-agent-template")


dotpromptinstructions = open('prompt-dot.md','r').read()
dotrdf = open('prompt-rdf.md','r').read()
dotsummarize =open('prompt-summarize-scholarly.md','r').read()


template = """
SYSTEM: {system}
CONTENT: {content}
"""

outtemplate = """
FORMAT: {format_instructions}
SYSTEM: {system}
CONTENT: {content}
"""

prompt = ChatPromptTemplate.from_template(template=template)
parser = JsonOutputParser(pydantic_model=Output)
outprompt = PromptTemplate(template=outtemplate,input_variables=["system","content","format_instructions"])
outspecificprompt = PromptTemplate(template=outtemplate,input_variables=["system","content"],partial_variables={"format_instructions":parser.get_format_instructions()})

chain = prompt | model
rdfchain = outspecificprompt | model | parser
dotchain = outprompt | model 

url = 'https://en.wikipedia.org/wiki/Metacognition'

pdffile = '/home/ccarthen/Documents/personalprojects/academic/3366423.3380259.pdf'
loader = PyPDFLoader(pdffile)
# Fetch the page content
response = requests.get(url)

# Parse the content using BeautifulSoup
soup = BeautifulSoup(response.content, 'html.parser')

# Extract the main content of the page
content = soup.find(id='bodyContent').get_text()
content = loader.load_and_split()
content = " ".join(list(map(lambda page: page.page_content, content)))
open('pdf_output.txt','w').write(content)
#print(type(content[0]))
#input()

content2 = chain.invoke({"system":dotsummarize, "content": content})
open('summarize_output.txt','w').write(content2)
print('Summarize Output')
input()
content3 = chain.invoke({"system":dotrdf, "content": content2})
open('rdf_output.txt','w').write(content3)

content4 = dotchain.invoke({"system":dotpromptinstructions, "content": content3,"format_instructions":"please only output the dot file and no elaboration.\n"})


open('graphviz_output.dot','w').write(content4)
source = Source(content4)
source.render('graphviz_output', format='png')
