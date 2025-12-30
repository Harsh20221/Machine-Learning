
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
""" print(docs) """
##* Splitting the Charecters
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
documents=text_splitter.split_documents(docs)
###* Embeddinng the Splitted Chunks
from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings()##?Initializing the embedder 
##* Storing the embeddings in a Vector Store
from langchain_community.vectorstores import FAISS
vectorstoredb=FAISS.from_documents(documents,embeddings) ##? This will take the splitted content and apply the embeddings and store these embeddings in the vector database 
print(vectorstoredb)

####-------THE ABOVE CODE IS COPIED FROM WEbbased.py file as the above code is used to setup the vector store db , Now after this in This Module , We'll be quering from the created db 

""" query="Langsmith has Two usage limits : Total Traces and extended " ###? We are simply running a query based on the saved data in the database 
result=vectorstoredb.similarity_search(query)##? Similarity search to find context reatable to query 
print(result[0].page_content) """
###* Making A retreival Chain



