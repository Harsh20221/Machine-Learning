import os
###* THE bELOW tWO LINES dISABLE some Warningg regarding tensorflow  
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
from dotenv import load_dotenv
load_dotenv()
##* This also Disables some warning 
os.environ.setdefault("USER_AGENT", "Mozilla/5.0 (compatible; LangChainBot/1.0)")
##* Initializing The Groq APi key
groq_api_key=os.getenv("GROQ_API_KEY")
##* InitiaLIZING the Higgingface api key
os.environ["HF_TOKEN"]=os.getenv("HUGGINGFACE_API_KEY")
##* INITIALIZING THE Embedding Model to convert the extracted information to vectors 
from langchain_huggingface import HuggingFaceEmbeddings
embeddings =HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
##* iNITIALIZING groq LIBRARIES 
from langchain_groq import ChatGroq
llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant") 
from langchain_chroma import Chroma # type: ignore
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter

import bs4
loader=WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content","post-title","post-header")
        )
    )
)
docs=loader.load()

##* Since Every LLM has a context size so we'll divide this document and break it down to smaller chunks THAT We obtrained from the webpage above 
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)##? Sertting Up text splitter to divide and create the
splits=text_splitter.split_documents(docs)
###* Now Storing all the vector embeddings to the chroma database so that later we can also apply similarity search 
vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)##? Initializing our vectorstore
result=vectorstore.similarity_search("CAT") ##?Applying Similarity Search
""" print(result) """

###*Turning a search function into a LangChain runnable
 ###?A runnable object is just a standard LangChain wrapper around a function.
##?A runnable object is just a standard LangChain wrapper around a function.
##?In simple terms
##?our original function is:
##?vectorstore.similarity_search(...)
##?That is just a normal Python method.
##?When you do:RunnableLambda(vectorstore.similarity_search) --wE are telling LangChain:
#/“Treat this function like a LangChain component”
#/ so it can use methods like:
#/  .invoke()
#/  .batch()
#/  .stream()
#/Without wrapping it, it is just a function.
#/With the wrapper, LangChain can handle it in a consistent way inside chains and pipelines.
###Analogy
#?Think of it like this:
#?normal function = a regular car
#?runnable = the same car, but now it has a standard GPS, dashboard, and controls that fit into LangChain’s system
from  typing import List
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
retreiver=RunnableLambda(vectorstore.similarity_search).bind(k=1)
retreiver.batch(["Cat","Dog"])
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

message="""
ANSWER tHIS qUESTION Using the provided Context Only {question} Context:{context}
"""

prompt=ChatPromptTemplate.from_messages([("human",message)])

ragchain={"context":retreiver,"question":RunnablePassthrough()} | prompt| llm
response=ragchain.invoke("Tell Me About Dogs")
print(response.content)
