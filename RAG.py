
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import numpy as np
import faiss
import nltk
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('punkt')
from nltk.tokenize import sent_tokenize


with open("cleaned_text.txt", "r", encoding="utf-8") as f:
    text_content = f.read()


lines = [line.strip() for line in text_content.splitlines() if line.strip()]
chunk_size = 10
chunks = [" ".join(lines[i:i+chunk_size]) for i in range(0, len(lines), chunk_size)]


model_embed = SentenceTransformer("intfloat/multilingual-e5-large")
corpus_embeddings = np.vstack([
    model_embed.encode(text, normalize_embeddings=True) for text in chunks
])


index = faiss.IndexFlatIP(corpus_embeddings.shape[1])
index.add(corpus_embeddings)


qa_model_name = "deepset/xlm-roberta-base-squad2"
qa_pipeline = pipeline("question-answering", model=qa_model_name, tokenizer=qa_model_name)


app = Flask(__name__)


short_term_memory = []


def compute_groundedness(answer, context):
    return answer in context

def compute_relevance(query_embedding, retrieved_embeddings):
    sims = cosine_similarity(query_embedding, retrieved_embeddings)
    avg_sim = float(np.mean(sims))
    return round(avg_sim, 4)


def generate_answer(query, k=6):
    memory_context = "\n".join([f"Q: {q}\nA: {a}" for q, a in short_term_memory[-3:]])
    query_embedding = model_embed.encode([query], normalize_embeddings=True)
    D, I = index.search(np.array(query_embedding), k)
    retrieved_chunks = [chunks[i] for i in I[0] if i < len(chunks)]
    retrieved_context = "\n".join(retrieved_chunks)
    full_context = memory_context + "\n" + retrieved_context

    try:
        result = qa_pipeline(question=query, context=full_context)
        answer = result["answer"].strip()
        grounded = compute_groundedness(answer, retrieved_context)
        relevance = compute_relevance(query_embedding, corpus_embeddings[I[0]])
        short_term_memory.append((query, answer))
        return {
            "answer": answer,
            "grounded": grounded,
            "relevance": relevance,
            "top_chunks": retrieved_chunks
        }
    except Exception as e:
        return {
            "answer": f"উত্তর খুঁজে পাওয়া যায়নি (ত্রুটি: {e})",
            "grounded": False,
            "relevance": 0.0,
            "top_chunks": []
        }

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    query = data.get("query")
    if not query:
        return jsonify({"error": "No query provided"}), 400

    result = generate_answer(query)
    return jsonify({
        "query": query,
        "answer": result["answer"],
        "grounded": result["grounded"],
        "relevance": result["relevance"],
        "chat_history": short_term_memory[-5:],
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    app.run(debug=True)
