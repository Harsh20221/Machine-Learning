

###* Here we will be splitting the HTML into chunks
from langchain_text_splitters import HTMLHeaderTextSplitter
html_string="""<!DOCTYPE html>
<html>
<body>
    <div>
        <h1>Foo</h1>
        <p>Some intro text about Foo.</p>
        <div>
            <h2>Bar main section</h2>
            <p>Some intro text about Bar.</p>
            <h3>Bar subsection 1</h3>
            <p>Some text about the first subtopic of Bar.</p>
            <h3>Bar subsection 2</h3>
            <p>Some text about the second subtopic of Bar.</p>
        </div>
        <div>
            <h2>Baz</h2>
            <p>Some text about Baz</p>
        </div>
        <br>
        <p>Some concluding text about Foo</p>
    </div>
</body>
</html>
"""
headers_to_split_on=[
    ("h1","Header1"),
    ("h2","Header2"),
    ("h3","Header3"),  
]

html_splitter=HTMLHeaderTextSplitter(headers_to_split_on)
html_header_splits=html_splitter.split_text(html_string)
""" print(html_header_splits) """

###* Here we will also be splitting HTML into chunks by using URL instead of Provided HTML data
url='https://plato.stanford.edu/entries/goedel/'
html_splitter2=HTMLHeaderTextSplitter(headers_to_split_on) ##? Same as the above example just repeated for visualization purposes
html_header_splits2=html_splitter2.split_text_from_url(url)
""" print(html_header_splits2) """



###* Here we will be splitting Json into Chunks  by using an API
import json
import requests
json_data=requests.get("https://api.smith.langchain.com/openapi.json").json()
from langchain_text_splitters import RecursiveJsonSplitter
json_splitter=RecursiveJsonSplitter(max_chunk_size=100)
json_chunks=json_splitter.split_json(json_data)
""" for chunk in json_chunks:
    print(chunk) """
    
###* The splitter can also output Documents 
docs=json_splitter.create_documents(texts=[json_data])
for doc in docs[:3]:
    print(doc)
    

