# %%
from dotenv import load_dotenv
import openai
import requests
from openai import OpenAI
import json
from pprint import pprint
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm
load_dotenv() 

# %% [markdown]
# ## Data Loadiing Part

# %%
docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()

documents = []

for course in documents_raw:
    course_name = course['course']

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# %%
pprint(documents[1])

# %%

es_client = Elasticsearch(
    "http://localhost:9200",
    # forces 8-compatible headers
)

# %%
index_settings = {
    "settings": {
        "number_of_shards": 1,
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "section": {"type": "text"},
            "question": {"type": "text"},
            "course": {"type": "keyword"} 
        }
    
}
}
index_name = "course-questions"

es_client.indices.create(index=index_name, body=index_settings)

# %%
for doc in tqdm(documents):
    es_client.index(index=index_name, document=doc)

# %%
query = 'How do copy a file to a Docker container?'



# %%
def elastic_search(query):
    search_query = {
    "size": 3,
    "query": {
        "bool": {
            "must": {
                "multi_match": {
                    "query": query,
                    "fields": ["question^4", "text"],
                    "type": "best_fields"
                }
        },
            "filter": {
                "term": {
                    "course": "machine-learning-zoomcamp"
                    }
                }
            }
        }
    }
    search_response = es_client.search(index=index_name, body=search_query)
    result_docs = []

    for hit in search_response['hits']['hits']:
        result_docs.append(hit['_source'])

    return result_docs

# %% [markdown]
# ### Retriving seaarch results(brute forced only top-3 results based on score here. Now to add the results as a context, question and prompt to the LLM

# %% [markdown]
# 

# %%
def build_prompt(query, search_results):
    template = (
        "You're a course teaching assistant. Answer the QUESTION based on the CONTEXT "
        "from the FAQ database. Use only the facts from the CONTEXT when answering.\n\n"
        "QUESTION: {question}\n\nCONTEXT:\n{context}"
    )

    # Efficiently assemble the context blocks
    blocks = [
        f"section: {doc['section']}\n"
        f"question: {doc['question']}\n"
        f"answer: {doc['text']}"
        for doc in search_results
    ]
    context = "\n\n".join(blocks)

    return template.format(question=query, context=context)

# %%
client = OpenAI()
def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

# %%
def rag(query):
    search_results = elastic_search(query)
    prompt = build_prompt(query, search_results)
    pprint(prompt)
    print(len(prompt))
    answer = llm(prompt)
    return answer
prompt= rag(query)
prompt

# %%
prompt

# %%
## Calculating the tokens - 
import tiktoken
encoding = tiktoken.encoding_for_model("gpt-4o")
len(encoding.encode(prompt))  # 2048 tokens

# %%
encoding.decode_single_token_bytes(63842)

# %%



