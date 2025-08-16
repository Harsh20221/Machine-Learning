corpus="""This Building belongs to Microsoft enterprizes but the lake behind belongs to emmar real estate developers"""

import nltk
# First tokenize the text into words
tokens = nltk.word_tokenize(corpus)
# Then do POS tagging on the tokens
pos_tags = nltk.pos_tag(tokens)
print(pos_tags)
nltk.download('words')
nltk.download('maxent_ne_chunker_tab')
tag_elements=nltk.ne_chunk(pos_tags)
print(tag_elements)