import os
from dotenv import load_dotenv
load_dotenv()
###* Loading the API
groq_api_key=os.getenv("GROQ_API_KEY")

##* Importing the groq chat 
from langchain_groq import ChatGroq
##* Initializing Model 
model=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)
#*ACTUal code
from langchain_core.messages import HumanMessage,SystemMessage #? To specify the llm whcch message is provided by human being and which is a system message , System message Instructs the LLM how really it needs to behave or work
messages=[
    SystemMessage(content="Translate the Following From English to French "),
    HumanMessage(content="Hello How are you ")
]
result = model.invoke(messages)
print(result.content)

