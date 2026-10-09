from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

facts = [
    "BlueShoe was founded in 2019.",
    "We sell eco-friendly shoes made from recycled plastic.",
    "Our shoes come in red, blue, and green.",
]

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_db = FAISS.from_texts(facts, embeddings)
print("FAISS database built (running locally)!")

results = faiss_db.similarity_search("What colors are available?", k=1)
print("Best match:", results[0].page_content)