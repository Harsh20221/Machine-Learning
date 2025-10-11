import os
from dotenv import load_dotenv

# load .env variables into environment
load_dotenv()

# read API key from env (do not assign None into os.environ)
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key is None:
	raise SystemExit(
		"OPENAI_API_KEY is not set. Add it to your environment or a .env file (e.g. OPENAI_API_KEY=sk-...)"
	)

# if present, ensure it's available in os.environ (it's already there from dotenv/load)
os.environ["OPENAI_API_KEY"] = openai_key
##* Here we are doing embeddings using Open AI api embedder , basically we are convertuing text into vectors
from langchain_openai import OpenAIEmbeddings
embeddings=OpenAIEmbeddings(model="text-embedding-3-large")
text="This is an example text for Open AI embeddings"
query_result=embeddings.embed_query(text)
print(query_result)
###* Storing the generated Vectors from the API 
from langchain_community.vectorstores import Chroma
db=Chroma.from_documents()