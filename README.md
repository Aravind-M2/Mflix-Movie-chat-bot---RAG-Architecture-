# 🎬 RAG-Based Question Answering System Using MongoDB Movies Dataset  

This project is an intelligent system designed to answer user queries about movies by leveraging a **Retrieval-Augmented Generation (RAG)** architecture. It involves migrating data from a NoSQL database (MongoDB) to a structured SQL database and then using that data to build a prompt-based QA system.

---

## 🚀 Key Takeaways  

- **Data Engineering:** Perform MongoDB → SQL data migration using Python, design a relational schema, and preprocess data for a structured format.  
- **RAG Architecture:** Understand and implement a RAG architecture using the LangChain framework.  
- **LLM Integration:** Integrate a Large Language Model (LLM) like OpenAI’s GPT with structured data for intelligent response generation.  
- **Vector Search:** Create and utilize a vector store (FAISS) for efficient semantic search.  

---

## 🏷️ Domain  
Media & Entertainment / AI-Powered Chat Systems  

---

## 📝 Problem Statement  
Design and develop a system that can answer user queries based on the `sample_mflix` movie dataset.  
The system’s core functionality relies on a **RAG architecture** that uses structured data from a SQL database, populated by migrating data from MongoDB.

---

## 💡 Business Use Cases  

- AI chat assistant for movie recommendation platforms.  
- Intelligent FAQ system for cinemas or streaming services.  
- Semantic search for media metadata and user feedback.  
- Generate insights from user reviews and ratings.  

---

## ⚙️ Approach & Technical Stack  

The project is divided into two main phases: **Data Engineering (Phase 1)** and **RAG System Development (Phase 2).**

### Phase 1 – Data Migration & Preprocessing  
- **Extract:** Python scripts connect to and extract data from the `sample_mflix` dataset in MongoDB.  
- **Transform:** Nested documents are flattened and collections are normalized to fit a relational schema.  
- **Load:** The transformed data is pushed into a PostgreSQL database.  

### Phase 2 – RAG System  
- **Embedding:** Combine relevant fields (movie plots, reviews) and vectorize them using OpenAI embeddings.  
- **Vector Store:** Store the generated embeddings in a **FAISS** vector store for fast similarity search.  
- **RAG Logic:** Implement retrieval logic with **LangChain**, fetching relevant text chunks from the vector store based on a user’s query.  
- **Response Generation:** Use an LLM (OpenAI GPT-3.5) to generate accurate, context-aware, and intelligent responses.  

---

## 🛠️ Technologies Used  

- **Databases:** MongoDB, PostgreSQL  
- **Programming:** Python  
- **Frameworks/Libraries:** LangChain, OpenAI, FAISS, Streamlit, Pandas, psycopg2, SQLAlchemy, PyMongo, python-dotenv  

---

## 📂 Project Structure  

| File / Folder | Description |
|---------------|-------------|
| `Version1(phase1.2).ipynb` | Connect to MongoDB, explore collections, export data to CSV. |
| `Version1(phase1.3).ipynb` | Inject pre-processed CSV data into PostgreSQL. |
| `Version1(phase2.2) copy.ipynb` | Demonstrates the RAG build process, including creating embeddings. |
| `working_app_v3.py` | Main Streamlit application providing the user interface for the RAG-based QA system. |
| `faiss_movie_index/` | Directory containing the FAISS vector store files. |
| `.env` | API keys and database credentials. |

---

## ⚡ Setup & Installation  

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd <your-project-folder>
