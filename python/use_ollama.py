import json
import chromadb
import ollama
from chromadb.utils import embedding_functions

embedding_model = "mxbai-embed-large"

lang="ja"

if lang == "en":
    documents = [
    "Llamas are members of the camelid family meaning they're pretty closely related to vicuñas and camels",
    "Llamas were first domesticated and used as pack animals 4,000 to 5,000 years ago in the Peruvian highlands",
    "Llamas can grow as much as 6 feet tall though the average llama between 5 feet 6 inches and 5 feet 9 inches tall",
    "Llamas weigh between 280 and 450 pounds and can carry 25 to 30 percent of their body weight",
    "Llamas are vegetarians and have very efficient digestive systems",
    "Llamas live to be about 20 years old, though some only live for 15 years and others live to be 30 years old",
    ]
elif lang == "ja":
    documents = [
    "リャマはラクダ科の一員で、ビクーニャやラクダと非常に近い関係にあります",
    "リャマは4,000年から5,000年前にペルーの高地で最初に家畜化され、荷物運搬動物として使用されました",
    "リャマは最大で6フィート（約1.8メートル）の高さまで成長しますが、平均的なリャマは5フィート6インチから5フィート9インチ（約1.68メートルから1.75メートル）の高さです",
    "リャマの体重は280から450ポンド（約127から204キログラム）で、自身の体重の25から30パーセントを運ぶことができます",
    "リャマは草食動物で、非常に効率的な消化系を持っています",
    "リャマの寿命は約20年ですが、15年で死ぬものもいれば、30年まで生きるものもいます",
    ]

client = chromadb.Client()
collection = client.get_or_create_collection(name="collection_name")


# You can also use the Ollama API to get embeddings via curl:
# curl http://localhost:11434/api/embeddings -d '{
#   "model": "mxbai-embed-large",
#   "prompt": "Llamas are members of the camelid family"
# }'

for i, d in enumerate(documents):
    response = ollama.embeddings(model=embedding_model, prompt=d)
    embedding = response["embedding"]
    collection.add(
        ids=[str(i)],
        embeddings=[embedding],
        documents=[d]
    )

if lang == "en":
    prompt = "Do llamas hunt for meat?"
elif lang == "ja":
    prompt = "リャマは肉を狩りますか？"

response = ollama.embeddings(
    prompt=prompt,
    model=embedding_model
)

results = collection.query(
    query_embeddings=[response["embedding"]],
    n_results=1
)
data = results['documents'][0][0]

output = ollama.generate(
    model="llama2",
    prompt=f"Using this data: {data}, Respond to this prompt: {prompt}"
)

print(output['response'])