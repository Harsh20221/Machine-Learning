import os
try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv()
groq_api_key=os.getenv('GROQ_API_KEY')
for proxy_var in (
    'ALL_PROXY',
    'all_proxy',
    'HTTPS_PROXY',
    'https_proxy',
    'HTTP_PROXY',
    'http_proxy',
    'FTP_PROXY',
    'ftp_proxy',
    'GRPC_PROXY',
    'grpc_proxy',
):
    os.environ.pop(proxy_var, None)

##* Importing the groq chat 
from langchain_groq import ChatGroq
##* Initializing Model 
model=ChatGroq(model="llama-3.1-8b-instant",groq_api_key=groq_api_key)


#* Creating template for the Chat 
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

model.invoke(
    [
        HumanMessage(content="Hi, my name is hARSH"),AIMessage(content="Hi harsh , how are you ?")
    ]
)
#* Enabling Message History
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
#? Once we give a session id this function below wll check if the id is in "store" dictionary and check whatever chat history is there for a specific session id
store={}###? This dictionary will store the session id 
#/bELOW tHIS IS THE MAIN FUNCTION THAT MANAGES tHE STORINGF OF  OLD USER  MESSAGES AND RETREIVAL OF OLD MESSAGES AND HELPS AI REMEMBER STUFF 
def get_session_history(session_id:str)->BaseChatMessageHistory: #? The return type of this function will be BasecHATmESSAGEhISTORY
    if session_id not in store:
        store[session_id]=ChatMessageHistory() ###? This is an object of a chatmesaagehistory , WHtever chat is happening it'll be directly going to the session id 
    return store[session_id]
with_message_history=RunnableWithMessageHistory(model,get_session_history)    #? ANALOGY-- store = memory box
#?session_id = user’s folder name
#?ChatMessageHistory = empty notebook
#?RunnableWithMessageHistory = helper that attaches the notebook to the model



config={"configurable":{"session_id":"chat1"}} ###?session_id = the name of the chat
##?config = the instruction packet that carries that name to the system

###?Here we are chatting with the ai and we are also passing a config that stores the session id , Using this the ai can remember the things we say to it         
response = with_message_history.invoke(
    [HumanMessage(content="Hi my name is Harsh")], config=config
)

###? Hence since we passed the same config as earlier then the ai will remeber our chats ,it'll say it remeber our name866
config1={"configurable":{"session_id":"chat1"}}
response=with_message_history.invoke(
    [HumanMessage(content="What's My Name do you remember my name")],config=config1
)

response.content#! add print to print the output , Here It's removed to avoid multiple prints 

##L-2------------------------------

####* wORKING WITH CHAT PROMPT TEMPLATE AND With message history altogether
#?At runtime, this app does this every time the user sends something:

#?User types one new message.
#?You wrap that text in a HumanMessage.
#?RunnableWithMessageHistory fetches the old chat history for that session.
#?It combines old messages plus the new HumanMessage.
#?The prompt template inserts that whole list at the MessagesPlaceholder.
#?The model answers.
#?The wrapper saves both the user message and the AI reply back into history.

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful Assistant , Answer Alll the questions to the best of your ability"),
        MessagesPlaceholder(variable_name="messages") #?A good mental model is:

#?MessagesPlaceholder = empty container
#?messages = the list that fills it
    ] ###?Here we are using a message placeholder called as:--- variable_name=messages , earlier we were using  aiMESSAGE , hUMAN message and all 
)
 
chain=prompt|model ##? Here we are creating a chain

chain.invoke({"messages":[HumanMessage(content="Hi my name IS haRsh")]}) #?The placeholder is just the empty slot in the prompt. It does not get the first message when you create it. It gets filled only when you run the chain with chain.invoke and pass the messages list.

with_message_history=RunnableWithMessageHistory(chain,get_session_history) ##?that one human message above inside  chain.invoke is the new question. The RunnableWithMessageHistory wrapper then does the rest:

#?it fetches earlier messages from the session history it combines those past messages with the new HumanMessage it sends the full list into the prompt where MessagesPlaceholder(variable_name="messages") is located after the model answers, it stores both the new human message and the new AI message back into history
 
config3={"configurable":{"session_id":"chat3"}}

response=with_message_history.invoke([HumanMessage(content="Hi I like Pizza")],config=config3)

#* Add MORE Complexity but here running without message history


prompt1=ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful Assistant , Answer Alll the questions to the best of your ability in {language}"),
        MessagesPlaceholder(variable_name="messages")


    ] 
)

chain=prompt1|model
response1=chain.invoke({"messages":[HumanMessage("Hi my name is Harsh")],"language":"Hindi"}) ##? Here we are also passing a second parameter called as message's Language 
""" print(response1.content) """


##* Passing / expereminting with above complexities along with message history 
with_message_history=RunnableWithMessageHistory(
    chain,get_session_history, ##?Here we are using the old chain we created with our old prompt termplate instead of creating a new prompt template
    input_messages_key="messages"
)

confignew={"configurable":{"session_id":"chat4"}}

response=with_message_history.invoke(
    {"messages":[HumanMessage(content="hI i AM hARsh")],"language":"Hindi"},config=confignew ##!!the config will not be inside the messages and will be outside the messages while the language and human message will be inside the messages 
)
""" print(response.content) """

##* Managing the Conversation History so that the messages does not overflow the context window of the llm --We'll use trimmessages

from langchain_core.messages import SystemMessage, trim_messages
trimmer=trim_messages(
    max_tokens=40,
    strategy="last",#?By mentioning last , the Trimmer will count from the last message it received
    token_counter="approximate",
    include_system=True,#?By saying include_system = True we are saying or telling it top also count the system messages
    allow_partial=False,#?Since we do not want partial information so we make allow_partial to false
    start_on="human"#?We also mention from where it needs to start , Here we want it to start from human messages 
    #!!The empty list came from the trimmer config, not from the message data. In this file, start_on was using "Human"; this LangChain version expects lowercase "human"
    
)

##* tESTING THE tRIMMER MODEL BY Using a demo conversation 
messages=[
    HumanMessage(content="Hi I am harsh "),
    AIMessage(content="Hi Harsh How are You?"),
    HumanMessage(content="i AM FINE "),
    AIMessage(content="Nice"),
    HumanMessage(content="I like ice Cream ") ,
    AIMessage(content="That's Nice"),
    HumanMessage(content="I love You  "),
    AIMessage(content="Nice Me too"),
    HumanMessage(content="I like you like Ice Cream"),
    AIMessage(content="I like ice cream too")
]

output=trimmer.invoke(messages)##? Change the "max_tokens"(line-143) to allow how much part of messages to trim 

print(output)

