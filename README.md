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
| `Version1(phase2.2).ipynb` | Demonstrates the RAG build process, including creating embeddings. |
| `working_app_v3.py` | Main Streamlit application providing the user interface for the RAG-based QA system. |
| `faiss_movie_index/` | Directory containing the FAISS vector store files. |
| `.env` | API keys and database credentials. |

---

## ⚡ Setup & Installation  

### 1. Clone the repository
```bash
git clone <https://github.com/Aravind-M2/Mflix-Movie-chat-bot---RAG-Architecture-.git>
cd <your-project-folder>

## 2. Set up the Databases

- Ensure you have a **MongoDB Atlas** account with the `sample_mflix` dataset loaded.  
- Set up a local **PostgreSQL** database.  

Create a `.env` file in the project root with your credentials:

```env
OPENAI_API_KEY="your_openai_api_key"
DB_USER="postgres"
DB_PASSWORD="your_db_password"
DB_HOST="localhost"
DB_PORT="5432"
DB_NAME="mflix"

## 3. install the required packages with

pip install -r requirements.txt

## 4. Run the ETL Process

Execute the Jupyter notebooks Version1(phase1.2).ipynb and Version1(phase1.3).ipynb in order to migrate the data from MongoDB to your PostgreSQL database.

## 5. Create the FAISS Index

Run the script/notebook that creates the FAISS vector store ( Version1(phase2.2).ipynb)
This will generate the faiss_movie_index directory.

## 6. Run the Application
streamlit run working_app_v3.py

**### Usage**

Once the Streamlit application is running, you can interact with the chat interface to ask questions about the movies in the dataset.

The system will classify your query and provide a response based on either:

Structured database lookup (SQL)

Semantic search using the RAG model (FAISS + LLM)
