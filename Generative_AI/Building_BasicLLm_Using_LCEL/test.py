import os
from dotenv import load_dotenv
load_dotenv()
#loading the api
groq_api_key=os.getenv("GROQ_API_KEY")
from langchain_groq import ChatGroq
model=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)
from langchain_core.messages import HumanMessage,SystemMessage
messages=[
    SystemMessage(content="Translate to Russian"),HumanMessage(content="I live in tokyo")
]
result=model.invoke(messages)
from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()
parser.invoke(result)

chain=model|parser
chain.invoke(messages)


from langchain_core.prompts import ChatPromptTemplate
prompt=ChatPromptTemplate.from_messages(ssystem='')





























