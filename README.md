# 📚 Multilingual RAG System — Bengali-English QA from PDF Corpus

This project is a Level-1 AI Engineer assessment solution for building a **Retrieval-Augmented Generation (RAG)** system capable of answering English and Bengali queries from the Bengali short story _"অপরিচিতা"_ by Rabindranath Tagore (sourced from the HSC26 PDF).

---

## 🚀 Features

- Multilingual query support: English & Bengali
- OCR-based PDF parsing (Bengali language)
- Sentence-based document chunking
- Embedding + FAISS vector store for similarity search
- RAG-based QA using XLM-RoBERTa
- REST API for querying
- Evaluation: relevance score + groundedness check

---

## 🛠️ Tools & Libraries Used

| Purpose                        | Tool/Library                                 |
|-------------------------------|----------------------------------------------|
| OCR                           | `pytesseract`, `pdf2image`, `Pillow`         |
| Text preprocessing            | `nltk`                                       |
| Embeddings                    | `sentence-transformers` (`intfloat/multilingual-e5-large`) |
| Vector storage                | `faiss-cpu`                                  |
| Question answering (QA)       | `transformers` (`deepset/xlm-roberta-base-squad2`) |
| REST API                      | `flask`                                      |

---

## 📁 Directory Structure

```
.
├── HSC26.pdf              # Raw scanned book PDF
├── extraction.py          # PDF OCR extraction script
├── cleaned_text.txt       # Cleaned extracted text
├── text.txt               # Raw OCR output
├── RAG.py                 # Main RAG system with REST API
├── Sample Questions.txt   # Example questions for testing
├── requirements.txt       # Required packages
```

---

## 🔧 Setup Instructions

1. **Install Tesseract & Bengali Language Pack**
   - Download from: https://github.com/tesseract-ocr/tesseract
   - Add to System Variables PATH.
   - Ensure Bengali language pack (`ben.traineddata`) is installed in `tessdata`.

2. **Install Poppler for `pdf2image`**
   - Windows: Download from https://github.com/oschwartz10612/poppler-windows/releases/
   - Add `bin/` folder to System Variables PATH.

3. **Install Python Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run OCR Extraction (optional)**
   ```bash
   python extraction.py  # Generates `text.txt`
   ```

5. **Run the API**
   ```bash
   python RAG.py
   ```

---

## 🧪 Sample Queries & Outputs

Try via `/ask` endpoint with (Tested using POSTMAN)
Example API call: http://127.0.0.1:5000/ask

```JSON Format
{
  "query": "অনুপমের পিসতুতো ভাইয়ের নাম কী?"
}
```

Response:
```JSON
{
    "answer": "বিনুদাদা,",
    "chat_history": [
        [
            "অনুপমের পিসতুতো ভাইয়ের নাম কী?",
            "বিনুদাদা,"
        ]
    ],
    "grounded": true,
    "query": "অনুপমের পিসতুতো ভাইয়ের নাম কী?",
    "relevance": 0.7864
}

```
Sample questions and generated answers::
```Questions:
{
Question: অনুপমের পিসতুতো ভাইয়ের নাম কী?
Answer: বিনুদাদা
Question: How old was Onupom?
Answer: সাতাশ
Question: কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?
Answer: মামা
Question: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?
Answer: শস্তুনাথবাবু
Question: রবীন্দ্রনাথ ঠাকুরের জন্ম কোথায়?
Answer: জোড়ার্সীকো, কলকাতা, ভারত।
Question: রবীন্দ্রনাথ ঠাকুরের জন্ম কত সালে?
Answer: ৭ মে, ১৮৬১
Question: অপরিচিতা কোন পত্রিকায় প্রথম প্রকাশিত হয়?
Answer: সবুজপত্র
Question: When did Rabindranath Tagore die?
Answer: ৭ আগস্ট, ১৯৪১
}

```

---

## 📊 Evaluation Metrics

| Metric        | Description                                               |
|---------------|-----------------------------------------------------------|
| Groundedness  | Checks if the answer is present in retrieved chunks       |
| Relevance     | Cosine similarity between query and retrieved embeddings  |

---

## 📌 Methodological Answers

### 1. **Text Extraction Method**
**Tool:** `pytesseract` + `pdf2image`  
**Reason:** The original PDF text used ANSI encoding. Therefore, OCR was used to extract text.  
**Challenge:** Libraries like PyMuPDF (a.k.a fitz) cannot be used for ANSI encoding. Had to switch to OCR for text extraction. 
               Lack of Bangla PDF parser and extraction tools were also a big challenge.

---

### 2. **Chunking Strategy**
**Type:** Sentence-based, with 10-line sliding window  
**Why:** Sentence tokenization preserves semantic boundaries, ensuring meaningful context during retrieval.

---

### 3. **Embedding Model**
**Used:** `intfloat/multilingual-e5-large`  
**Why:** It supports Bengali and English, and produces high-quality sentence embeddings optimized for semantic similarity. 
Other models like "l3cube-pune/bengali-sentence-similarity-sbert", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
were used with worse results.

---

### 4. **Similarity & Storage Method**
**Similarity:** Cosine Similarity  
**Storage:** FAISS (`IndexFlatIP`)  
**Why:** Efficient similarity search over normalized embeddings using inner product (equivalent to cosine).

---

### 5. **Meaningful Query-Chunks Comparison**
**Method:** 
- Use multilingual embeddings to align query with semantically related context.
- Combine short-term memory with retrieved chunks to give QA model broader context.

**Handling Vague Queries:** Model may degrade gracefully but can return generic or incomplete answers. Can be improved with:
- Better chunk granularity
- Advanced reranking
- Query expansion techniques

---

### 6. **Relevance of Results**
**Observation:** Most answers are relevant and contextually grounded.  
**Possible Improvements:**
- Finer-grained chunking (sentence-level)
- Adding titles/headings during preprocessing
- Using rerankers like Cohere ReRank or LlamaIndex tools

---

## 🌐 API Endpoints

| Method | Route     | Description                      |
|--------|-----------|----------------------------------|
| POST   | `/ask`    | Accepts a query, returns answer  |
| GET    | `/health` | Basic health check               |

---

## ✅ Bonus

- ✅ REST API implemented via Flask
- ✅ Evaluation via groundedness + relevance
- ⛔ No front-end GUI or LangChain used for this Level-1 demo

---

## 🧠 Future Work

- Add LangChain integration
- Use MongoDB or Chroma for scalable vector store
- Fine-tune QA model with domain-specific Bengali examples
