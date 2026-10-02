from sentence_transformers import SentenceTransformer
import faiss
import os
knowledge_folder = "knowledge_base"
documents = []
document_names = []
for filename in os.listdir(knowledge_folder):
    if filename.endswith(".txt"):
        file_path = os.path.join(knowledge_folder, filename)

        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(text)
        document_names.append(filename)

print("Number of documents:", len(documents))
chunks = []
chunk_names = []
for document, name in zip(documents, document_names):
    paragraphs = document.split("\n\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if paragraph:

            chunks.append(
                f"{name}: {paragraph}"
            )

            chunk_names.append(name)

print("Number of chunks:", len(chunks))
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = embedding_model.encode(chunks)
print("Embedding shape:", embeddings.shape)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
print("Number of vectors in index:", index.ntotal)
def retrieve_information(query):
    query_embedding = embedding_model.encode([query])
    distances, indices = index.search(
        query_embedding,
        k=1
    )
    retrieved_index = indices[0][0]
    return chunks[retrieved_index]
