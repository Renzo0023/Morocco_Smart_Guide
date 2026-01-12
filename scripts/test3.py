# scripts/debug_all_cities.py
from app.rag.vectorstore import get_retriever

retriever = get_retriever(k=200)

docs = retriever.invoke("tourisme maroc")

cities = {}

for d in docs:
    c = (d.metadata or {}).get("city")
    cities[c] = cities.get(c, 0) + 1

print("\nVilles présentes dans FAISS:")
for k,v in sorted(cities.items(), key=lambda x: -x[1]):
    print(k, ":", v)
