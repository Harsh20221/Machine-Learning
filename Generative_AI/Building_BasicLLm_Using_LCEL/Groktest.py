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
##* The parser can help extract the output-only ( The result of out Query ) from the full response returned by the Model
from langchain_core.output_parsers import StrOutputParser
parser=StrOutputParser()
parser.invoke(result) 
###* USING LCEL to chain the components (Basically to Run the above code where we described the model and initialized the parser all at once)

chain=model|parser
chain.invoke(messages)


##* Designing a Chat Template
from langchain_core.prompts import ChatPromptTemplate

generic_template="Translate the following into{language}"
prompt=ChatPromptTemplate.from_messages(
    [("system",generic_template),("user","{text}")]
)
output=prompt.invoke({"language":"Mandarin","text":"HELLO"})
chain=prompt|model|parser
chain.invoke({"language":"french","text":"Hello"})

