# RagMind - RAG Based AI Chatbot

RagMind is an intelligent Retrieval-Augmented Generation (RAG) chatbot that answers user queries based on documents using semantic search and Large Language Models.

---

## Features

- Document-based question answering
- RAG pipeline (Retrieval + Generation)
- Semantic search using vector database (ChromaDB)
- LLM-powered responses
- Frontend + Backend support

---

## Project Structure

```
RagMind/
│
├── wonders/
│   ├── Backend/
│   ├── Frontend/
│   ├── RAG/
│   │   ├── chunking.py
│   │   ├── ingest.py
│   │   ├── vector_store.py
│   │   ├── main.py
│   │   └── docs/
│   ├── requirements.txt
│
├── .gitignore
└── README.md
```

---

## Tech Stack

- Python 
- ChromaDB (Vector Database)
- Large Language Models (LLM)
- HTML, CSS, JavaScript

---

## How It Works (RAG Flow)

1. Load documents  
2. Split into chunks  
3. Convert into embeddings  
4. Store in vector database  
5. User asks a question  
6. Retrieve relevant chunks  
7. LLM generates final answer  

---

## How to Run

### Clone the repository

git clone https://github.com/vignes2006waran/RagMind.git

cd RagMind


### Install dependencies

pip install -r wonders/requirements.txt


### Run the project

python wonders/RAG/main.py


---

## Future Improvements

- UI enhancements  
- Deployment (Cloud)  
- API integration  
- Multi-document support  

---

## Author

**Vigneshwaran J S**

---

## Support

If you like this project, give it a ⭐ on GitHub!
