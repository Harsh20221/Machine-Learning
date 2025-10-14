
##* Loading the necessary environment Variables
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
##* Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"

##* Main Code
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model='gpt-4o')
""" print(llm) """
##* Testing the Model Response
""" result=llm.invoke("What is Generative-Ai?")
print(result) """
##* Testing model response with the Prompt Template(Prompt Template will  add additional prompts along with the user prompt , This is done to finetune the user requests)
from langchain_core.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_messages([ ##!Make sure you don't type format_messages instead of from_messages as this will create syntax error
    ("system","You are an Expert AI Engineer. Provide me answers based on the question"),
    ("user","{input}")
])

""" chain=prompt|llm """ ##? This means that we need to combine our prompt along with the llm(In llm we are specifying which model we plan on using and other options)
""" response=chain.invoke({"input":"Can you Tell me about Langsmith?"}) """
""" print(response) """
#*Str-output-Parser
from langchain_core.output_parsers import StrOutputParser
output_parser=StrOutputParser()
chain=prompt|llm|output_parser
response=chain.invoke({"input":"Can you Tell me about Langgraph?"}) 
print(response)




