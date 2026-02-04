import os
from dotenv import load_dotenv
load_dotenv()
###* Loading the API
groq_api_key=os.getenv("GROQ_API_KEY")

##* Importing the groq chat 
from langchain_groq import ChatGroq
##* Initializing Model 
model=ChatGroq(model="gemma:2b",groq_api_key=groq_api_key)
print(model)

