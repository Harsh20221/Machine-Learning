
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

##* Assigning the Model 
from langchain_openai import ChatOpenAI
llm=ChatOpenAI(model="gpt-4o")

###* Making A retreival Chain 
#?A retrieval chain is a LangChain component that combines retrieval and language model generation to answer questions based on your own documents or data.
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_template("""
                                        Answer The following question Based only on the provided context 
                                        <context>{context}
                                        </context>
                                        """
)
##?The document chain (created with create_stuff_documents_chain) is a specific type of chain that takes retrieved documents and combines them with a prompt to generate an answer from a language model.
document_chain=create_stuff_documents_chain(llm,prompt)
#?The Document class is a fundamental data structure in LangChain that represents a piece of text along with associated metadata. It's essentially a container that holds content (usually text) and optional metadata (like source information, timestamps, or any custom attributes you want to attach)
from langchain_core.documents import Document

document_chain.invoke({
    "input":"Langsmit has two Usage limits: Total Traces and Extended Traces",
    "context":[Document(page_content="Langsmith has two usage limits Total Traces and Extended")]
})
###* Initializing CVector Store as a Retreiver 
retriever=vectorstoredb.as_retriever()##?This line of code converts a vector store database into a retriever object, which is a specialized interface for searching and fetching relevant documents. Think of it as transforming a storage container (the vector store) into a search engine that can intelligently find the most relevant information based on queries.
from langchain.chains import create_retrieval_chain
retreival_chain=create_retrieval_chain(retriever,document_chain) #?This line creates a retrieval chain, which is a powerful LangChain component that orchestrates the entire process of retrieving relevant documents and generating answers based on those documents. Think of it as building a complete question-answering pipeline that connects two critical pieces: the search mechanism (retriever) and the answer generation mechanism (document_chain).

#* Getting the Response from the LLM
response=retreival_chain.invoke({"input": "Langchain has two input limits : Total traces and extended "})
print(response['answer'])