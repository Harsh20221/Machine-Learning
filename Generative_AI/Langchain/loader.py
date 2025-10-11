###* Load Text
from langchain_community.document_loaders import TextLoader
""" textloader=TextLoader('speech.txt')
text_documents=textloader.load() """
###*Load Pdf
from langchain_community.document_loaders import PyPDFLoader
""" loader=PyPDFLoader('attention.pdf')
docs=loader.load()
print(docs) """
##* Load Webpages 
from bs4 import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader(
    web_paths=["https://lilianweng.github.io/posts/2023-06-23-agent/"],
    bs_kwargs=dict(
        parse_only=SoupStrainer(class_=("post-title", "post-content", "post-header"))
    ),
)
docs=loader.load()
""" print(docs) """
##* Load Arxiv an Opensource library for 2 million articles related to science 
from langchain_community.document_loaders import ArxivLoader
loader=ArxivLoader(query='1605.08386',load_max_docs=2).load() ##? In the query parameter we enter the unique id for that particular article that we want to load 
print(loader)

