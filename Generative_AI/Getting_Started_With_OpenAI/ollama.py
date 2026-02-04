import os 
from langchain_community.llms import Ollama
from dotenv import load_dotenv
import streamlit as  st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser#??The StrOutputParser is a utility class that handles the conversion of LLM outputs into simple string format
load_dotenv()
##* LOADING API KEYS FOR LANGCHAIN
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_PROJECT"]=os.getenv("LANGCHAIN_PROJECT")
os.environ["LANGCHAIN_TRACING_V2"]="true"
##* Configuring the chat prompt template 
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a Helpful assistant, Please respond to the Question asked "),
        ("user","Question:{question}")
    ]
    
)

#*Streamlit Template 
st.title("Langchain Demo with OLLama opensource models")
input_text=st.text_input("What Question you have in your mind")

#*Ollama  LLama2 model
llm=Ollama(model="gemma:2b")##!!Make sure to write --model="" and not just the model name
output_parser=StrOutputParser()
chain=prompt|llm|output_parser ##? Under the hood, LangChain implements the | operator as a special method that creates a RunnableSequence. This means when you later call chain.invoke({"question": input_text}), the framework automatically passes the output of each stage as the input to the next stage.
if input_text:
    st.write(chain.invoke({"question":input_text})) ##? This will Invoke or run the model , Will act as an ignition after the user enters a response 
    





