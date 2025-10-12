
##* Initializing the Hugging Face Embeddings Api
import os 
from dotenv import load_dotenv
load_dotenv()
os.environ['LANGCHAIN_API_KEY']=os.getenv("LANGCHAIN_API_KEY")
###* performing the Hugging face embeddings powered by langchain 
from langchain_huggingface import HuggingFaceEmbeddings
embeddings=HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
test="This is a text document"
result=embeddings.embed_documents(test)
print(result)
    

