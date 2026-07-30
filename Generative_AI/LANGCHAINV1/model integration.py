from LEC1 import response
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ['OPENAI_API_KEY']= os.getenv('OPENAI_API_KEY')
os.environ['GOOGLE_API_KEY'] = os.getenv('GOOGLE_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

#INVOKE THE MODEL
from langchain.chat_models import init_chat_model ##?Instead of importing provider-specific classes (like from langchain_openai import ChatOpenAI or from langchain_anthropic import ChatAnthropic), 
##init_chat_model acts as a unified factory function that can instantiate chat models from almost any provider using a single function.
""" model=init_chat_model("gpt-4.1")
response=model.invoke("Hello How are you I like rtx 5080") """
""" print(response.content)  """
# Option 1: OpenAI (e.g. gpt-4o-mini or gpt-4o)
# model = init_chat_model("gpt-4o-mini")

##* initializing AND Using the Gemini API
model = init_chat_model("gemini-3.6-flash", model_provider="google_genai")

response = model.invoke("Hello How are you")
print(response.content)