import os
from  dotenv import load_dotenv
load_dotenv()
##* Initizalizing Huggingface
from langchain_huggingface import HuggingFaceEmbeddings
os.environ["HF_API_KEY"]=os.getenv("HUGGINGFACE_API_KEY")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
##* Initializing ChatGroq
from langchain_groq import ChatGroq
groq_api_key=os.getenv("GROQ_API_KEY")
llm=ChatGroq(
  groq_api_key=groq_api_key,model='llama-3.1-8b-instant'
)
##* Now Initializing the Web Based Loader along with beautifdul soup and loading the webpage
import bs4 
from langchain_community.document_loaders import WebBaseLoader 
loader=WebBaseLoader(web_path='https://lilianweng.github.io/posts/2023-06-23-agent/',
                     bs_kwargs=dict(
    parse_only=bs4.Soupstrainer(#? USIng BEautiful Soup's Soup Strainer to only load the required parameters
        class_=("post-content","post-title","post-header"))
))
docs=loader.load()
##* Now We'll do splitting to split the loaded webpage to smaller chunks 
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
splits=text_splitter.split_documents(docs) #!!Do not use split_text here , use split_documents as text is for small tgext fragments only 

###* Now Storing the EMBEDDINGS TO Vector STORE
from langchain_chroma import Chroma
vectorstore=Chroma.from_documents(documents=splits,embeddings=embeddings)
retreiver=vectorstore.as_retriever()

####*Now MAKING tHE  Prompt Template 
from langchain_core.prompts import ChatPromptTemplate

system_prompt=(
  " You are an AI assistant , Use the provided piece of context to answers the question asked by the user , If you dfon't know the answer say that you don't know how to do , Use 3 sentences max and keep the answers concise"
  "\n\n"
  "{context}"
)

prompt= ChatPromptTemplate.from_messages(
  [
    ("system",system_prompt),("human","{input}")
  ]
)


##* Creating a Rag Chain 
from lanchain-classic.chains.retrieval import create_retrieval_chains
from langchain.chains.combine_documents import create_stuff_documents_chain

question_answer_chain=create_stuff_document_chain(llm,prompt)
ragchain=create_retreival_chain(retreiver,question_answer_chain)
response=ragchain.invoke({"input":"What is Self Reflection"})

##* Using RetrievalQA
from langchain.chains import RetrievalQA

qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
answer = qa.run("What is Self Reflection")



