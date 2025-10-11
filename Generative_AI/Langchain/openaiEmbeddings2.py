
###* In this example we are not only making embeddings but we are also storing the embeddings in chroma Db
###* Loading the Dataset
from langchain_community.document_loaders import TextLoader
loader=TextLoader('speech.txt')
docs=loader.load()
##*Text Splitting the entire dataset 
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
final_documents=text_splitter.split_documents(docs)
##* Initializing the API KEY
import os
from dotenv import load_dotenv
load_dotenv()
os.environ["OPENAI_API_KEY"]=os.getenv("OPENAI_API_KEY")
###* Performing the Embeddings 
from langchain_openai import OpenAIEmbeddings
embeddings_1024=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=1024)
##* Vector Embedding And Vector StoreDB to store the embeddings (both in a single step) by embeddings we mean we are converting the text into vectors  and we are storing these vectors in chroma db which is a database to store vectors 
from langchain_community.vectorstores import Chroma
db=Chroma.from_documents(final_documents,embeddings_1024)
##* Running a Similarity search to check if the embeddings are stored successfully 
query="It is a distressing and oppressive duty, gentlemen of the Congress"
retrieved_results=db.similarity_search(query)
print(retrieved_results)

