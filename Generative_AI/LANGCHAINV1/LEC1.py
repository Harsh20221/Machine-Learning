import os
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']= os.getenv('OPENAI_API_KEY')


from langchain.agents import create_agent

def get_weather(city:str)->str:
    """get the weather of the city"""
    return f"The Weather for the {city} is Sunny"

agent=create_agent(
    model='gpt-5',
    tools=[get_weather],
    system_prompt="You are a good Assistant"
)


response=agent.invoke({"messages":[{"role":"user","content":"What is the Weather like in kolkata?"}]})
print(response['messages'][-1].content)