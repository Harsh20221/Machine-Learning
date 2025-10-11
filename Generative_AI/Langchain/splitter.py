from langchain_community.document_loaders import PyPDFLoader
loader=PyPDFLoader('attention.pdf')
docs=loader.load()

##* Recursively split  charecters
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
finaldocuments=text_splitter.split_documents(docs)
""" print(finaldocuments) """

####* Demo of Text splitting using Recursuive Text Splitter
speech=""
with open ("speech.txt") as f:
    speech=f.read()
text_split=RecursiveCharacterTextSplitter(chunk_size=100,chunk_overlap=20)
text=text_split.create_documents([speech])
""" print(text[0]) """
####* Demo of Text splitting using Charecter Text Splitter
from langchain_text_splitters import CharacterTextSplitter
text_split=CharacterTextSplitter(seperator='\n \n',chunk_size=100,chunk_overlap=20)
text_split.split_documents(docs)
print(docs)
    

