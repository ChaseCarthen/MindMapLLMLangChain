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
import re
import tiktoken
#from strip_tags import strip_tags



class Output(BaseModel):
    content: str = Field(description="The content where the output is stored.")


model = OllamaLLM(model="llama3.1",temperature=1,top_k=50,top_p=.5,repeat_penalty=1.2,num_ctx=16000)
#model = OllamaLLM(model="llama3.1:70b-instruct-q2_k")
#model = ChatOllama(model='llama3.1')


base_prompt = hub.pull("langchain-ai/react-agent-template")


dotpromptinstructions = open('prompt-dot.md','r').read()
dotrdf = open('prompt-rdf.md','r').read()
dotsummarize = open('prompt-summarize-scholarly.md','r').read()
#dotextractterms = open('prompt-extract-terms.md','r').read()
dotextractterms = open('prompt-terms.md','r').read()



template = """
SYSTEM: {system}
CONTENT: {content}
"""

summarize_template_2 = """
PAGE_INSTRUCTIONS: The page of the article is being provided to you. The summary section is your memory of what you have summarized so far in the article.
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

summarizeTemplate = """
SYSTEM: Combine these two contents to the best of your ability and retain information from both. Both of the contents are from summaries,relationships, and key ideas from differemt pages in a article.
CONTENT: {content}
CONTENT 2: {content2}
"""


def loadWebPage(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    content = soup.find(id='bodyContent').get_text()
    return content

def parsePDF(filename,all=False):
    loader = PyPDFLoader(filename)
    content = loader.load_and_split()
    if all:
        return " ".join(list(map(lambda page: page.page_content, content)))
    return content
def create_graph(chain,content):
    return content

def extract_terms(content):
    return content


sumprompt = ChatPromptTemplate.from_template(template=summarizeTemplate)
prompt = ChatPromptTemplate.from_template(template=template)
summarize_prompt = ChatPromptTemplate.from_template(template=summarize_template)
parser = JsonOutputParser(pydantic_model=Output)
outprompt = PromptTemplate(template=outtemplate,input_variables=["system","content","format_instructions"])
outspecificprompt = PromptTemplate(template=outtemplate,input_variables=["system","content"],partial_variables={"format_instructions":parser.get_format_instructions()})

chain = prompt | model
schain = sumprompt | model
summarize_chain = summarize_prompt | model
rdfchain = outspecificprompt | model | parser
dotchain = outprompt | model 

url = 'https://en.wikipedia.org/wiki/Metacognition'

pdffile = '2003.04761v1.pdf'
content = parsePDF(pdffile, all = False)


encoding = tiktoken.get_encoding('cl100k_base')
content2 = ''
pagesummaries = {}
summarylist = []
combinedSummaries = {"key_terms":[],"concepts":[],"relationships":[],"facts":[]}
combinedPages = {}

for index,page in enumerate(content):
    content2 = summarize_chain.invoke({"system":dotextractterms, "content": page, "page":index})
    pattern = re.compile(r'```json(.*?)```', re.DOTALL)

    # Find all matches
    matches = pattern.findall(content2)

    # Print the extracted code blocks
    for match in matches:
        content2 = match.strip()
    #print(content2)
    print(index,len(content))
    try:
        content2 =json.loads(content2)
        combinedSummaries["key_terms"] += content2["key_terms"]
        combinedSummaries["concepts"] += content2["concepts"]
        combinedSummaries["relationships"] += content2["relationships"]
        combinedSummaries["facts"] += content2["facts"]
        combinedSummaries["key_terms"] = list(set(combinedSummaries['key_terms']))
        print(len(encoding.encode(json.dumps(content2))))
        pagesummaries[f'page_{index}'] = content2
        combinedPages[f'page_{index}'] = content2

        if (index+1) % 4 == 0 and index > 0:
            summarylist.append(json.dumps(pagesummaries))
            pagesummaries = {}
            
    except Exception as e:
        print(e)
        continue

print('Generating GraphViz Output')
open('combinedout.txt','w').write(json.dumps(combinedPages))

for i,content in enumerate(summarylist):
    print(len(encoding.encode(content)))
    content4 = dotchain.invoke({"system":dotpromptinstructions, "content": content,"format_instructions":"please only output the dot file and please no elaboration.\n"})

    try:
        open(f'graphviz_output_{i}.dot','w').write(content4)
        source = Source(content4)
        source.render(f'graphviz_output_{i}.dot', format='png')
    except Exception as e:
        print(e)

print(len(encoding.encode(json.dumps(combinedSummaries))))
content4 = dotchain.invoke({"system":'Hi take this dataset and produce me a mindmap using only the DOT format from graphviz. Only output the DOT file please.', "content": json.dumps(combinedSummaries),"format_instructions":"please only output the dot file and please no elaboration.\n"})
#open('graphviz_output_combined.dot','w').write(content4)
source = Source(content4)
source.render('graphviz_output_combined.dot', format='png')