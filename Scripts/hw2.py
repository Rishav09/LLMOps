# %% [markdown]
# ### Q1: Embed the query: 'I just discovered the course. Can I join now?'. Use the 'jinaai/jina-embeddings-v2-small-en' model.You should get a numpy array of size 512.What's the minimal value in this array

# %%
from fastembed import TextEmbedding
model_handle = "jinaai/jina-embeddings-v2-small-en"

# %%
documents = "I just discovered the course. Can I join now?"

# %%
embedding_model = TextEmbedding(model_handle)
embeddings_generator = embedding_model.embed(documents)
embeddings_list = list(embeddings_generator)
len(embeddings_list[0])

# %%
import numpy as np
np.min(embeddings_list[0])

# %%
import numpy as np
np.linalg.norm(embeddings_list[0])

# %%
embeddings_list[0].dot(embeddings_list[0])

# %% [markdown]
# ### Q2 Cosine similarity with another vector: Now let's embed this document: doc = 'Can I still join the course after the start date?'

# %% [markdown]
# 

# %%
query_doc = 'Can I still join the course after the start date?'
query_embeddings_generator = embedding_model.embed(query_doc)
Qembeddings_list = list(query_embeddings_generator)
len(Qembeddings_list[0])

# %%
embeddings_list[0].dot(Qembeddings_list[0])

# %% [markdown]
# ## Ranking by cosine

# %%
big_documents = [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
  'section': 'General course-related questions',
  'question': 'How can we contribute to the course?',
  'course': 'data-engineering-zoomcamp'}]

# %%
type(big_documents)

# %%
texts = [
    f"{doc['text']}"
    for doc in big_documents
]
pprint.pprint(texts)

# %%
big_embeddings_generator = embedding_model.embed(texts)
embeddings_big = list(big_embeddings_generator)
len(embeddings_big[0])

# %%
doc_vectors = np.vstack(embeddings_big)

# %%
doc_vectors.shape

# %%
query_doc = 'Can I still join the course after the start date?'
query_vector = np.asarray(next(embedding_model.embed(query_doc)))
query_vector.shape

# %%
doc_vectors.dot(query_vector.ravel())

# %% [markdown]
# ### Q4: Ranking by cosine, version two full_text = doc['question'] + ' ' + doc['text']

# %%
new_texts = [
    f"{doc['question']} {doc['text']}"
    for doc in big_documents
]
pprint.pprint(new_texts)

# %%
big_embeddings_generator = embedding_model.embed(new_texts)
embeddings_big = list(big_embeddings_generator)
doc_vectors = np.vstack(embeddings_big)

# %%
doc_vectors.dot(query_vector.ravel())

# %%
## Selecting the embedding model

# %%
TextEmbedding.list_supported_models()

# %%
import json

EMBEDDING_DIMENSIONALITY = 384

for model in TextEmbedding.list_supported_models():
    if model["dim"] == EMBEDDING_DIMENSIONALITY:
        print(json.dumps(model, indent=2))

# %% [markdown]
# ## Q6: Indexing with Qdrant

# %%
import requests 

docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
docs_response = requests.get(docs_url)
documents_raw = docs_response.json()


documents = []

for course in documents_raw:
    course_name = course['course']
    if course_name != 'machine-learning-zoomcamp':
        continue

    for doc in course['documents']:
        doc['course'] = course_name
        documents.append(doc)

# %%
documents[:3]

# %%
model_handle = "BAAI/bge-small-en"
EMBEDDING_DIMENSIONALITY = 384

# %%
from qdrant_client import QdrantClient, models
client = QdrantClient("http://localhost:6333")

# %%
# Define the collection name
collection_name = "zoomcamp-hw2"
client.delete_collection(collection_name=collection_name)

# Create the collection with specified vector parameters
client.create_collection(
    collection_name=collection_name,
    vectors_config=models.VectorParams(
        size=EMBEDDING_DIMENSIONALITY,  # Dimensionality of the vectors
        distance=models.Distance.COSINE  # Distance metric for similarity search
    )
)

# %%
client.create_payload_index(
    collection_name=collection_name,
    field_name="course",
    field_schema="keyword"
)

# %%
points = []

for i, doc in enumerate(documents):
    text = doc['question'] + ' ' + doc['text']
    vector = models.Document(text=text, model=model_handle)
    point = models.PointStruct(
        id=i,
        vector=vector,
        payload=doc
    )
    points.append(point)

# %%
client.upsert(collection_name=collection_name, points=points)


# %%
q1 = "I just discovered the course. Can I join now?"


# %%
course = 'machine-learning-zoomcamp'
query_points = client.query_points(
    collection_name=collection_name,
    query=models.Document(
        text=q1,
        model=model_handle 
    ),
    query_filter=models.Filter( 
        must=[
            models.FieldCondition(
                key="course",
                match=models.MatchValue(value=course)
            )
        ]
    ),
    limit=5,
    with_payload=True
)
results = []
for point in query_points.points:
    results.append(point.payload)


# %%
pprint.pprint(query_points)

# %%



