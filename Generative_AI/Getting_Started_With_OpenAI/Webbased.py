
##* Loading the necessary environment Variables
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']=os.getenv('OPENAI_API_KEY')
##* Langsmith Tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
##*Loading Starts from here 
from langchain_community.document_loaders import WebBaseLoader
loader=WebBaseLoader("https://docs.langchain.com/langsmith/billing")
docs=loader.load()
print(docs)
