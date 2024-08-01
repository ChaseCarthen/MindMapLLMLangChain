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
import json
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

summarize_template = """
PAGE_INSTRUCTIONS: The page of the article is being provided to you. Please append to your summary under SUMMARY based on the new contents of the page and do not include the example I gave in the new summary
PAGE: {page}
SYSTEM: {system}
CONTENT: {content}
SUMMARY:
------------------------------- 
{summary}
"""

summarize_template = """
PAGE_INSTRUCTIONS: The page of the article is being provided to you
PAGE: {page}
SYSTEM: {system}
CONTENT: {content}
"""

outtemplate = """
FORMAT: {format_instructions}
SYSTEM: {system}
CONTENT: {content}
"""

prompt = ChatPromptTemplate.from_template(template=template)
summarize_prompt = ChatPromptTemplate.from_template(template=summarize_template)
parser = JsonOutputParser(pydantic_model=Output)
outprompt = PromptTemplate(template=outtemplate,input_variables=["system","content","format_instructions"])
outspecificprompt = PromptTemplate(template=outtemplate,input_variables=["system","content"],partial_variables={"format_instructions":parser.get_format_instructions()})

chain = prompt | model
summarize_chain = summarize_prompt | model
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
#content = " ".join(list(map(lambda page: page.page_content, content)))
wholecontent = " ".join(list(map(lambda page: page.page_content, content)))
open('pdf_output.txt','w').write(wholecontent)
#print(type(content[0]))
#input()
content = list(map(lambda page: page.page_content, content))
print(len(content))
#input()
content2 = ''
pagesummaries = {}
for index,page in enumerate(content):
    print(index)
    content2 = summarize_chain.invoke({"system":dotsummarize, "content": page, "page":index})
    pagesummaries[f'page_{index}'] = content2
    print(content2)
    if index == 8:
        break
content2 =  json.dumps(pagesummaries)
open('summarize_output.txt','w').write(content2)
print('Summarize Output')
input()
content3 = chain.invoke({"system":dotrdf, "content": content2})
open('rdf_output.txt','w').write(content3)

content4 = dotchain.invoke({"system":dotpromptinstructions, "content": content3,"format_instructions":"please only output the dot file and no elaboration.\n"})


open('graphviz_output.dot','w').write(content4)
source = Source(content4)
source.render('graphviz_output', format='png')
