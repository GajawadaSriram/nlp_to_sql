# AI-Powered NLP-to-SQL Agent

## 🚀 About The Project
This is a production-grade AI agent that converts natural language questions into executable SQL queries. It allows non-technical users to query relational databases securely without knowing SQL.

This project was built using a **Retrieval-Augmented Generation (RAG)** architecture to dynamically map user intent to database schemas while preventing AI hallucinations.

## 🛠️ Tech Stack
* **AI/LLM:** LangChain, HuggingFace/OpenAI models
* **Vector Database:** FAISS
* **Backend:** Python, Flask (or FastAPI)
* **Frontend:** React.js / HTML (Change to whatever you used)
* **Database:** SQLite / MySQL / PostgreSQL

## ✨ Key Features
* **RAG Pipeline:** Retrieves database schema context dynamically using FAISS to ensure accurate SQL generation.
* **Agentic Workflow:** The AI reasons through the user prompt, generates the SQL, executes it against the database, and returns the natural language answer.
* **Security Guardrails:** Validates SQL before execution to prevent malicious injection attacks.

## ⚙️ How It Works
1. User asks a question (e.g., "How many active users signed up this month?").
2. The LangChain agent fetches the relevant table schemas from the FAISS vector store.
3. The LLM generates a highly optimized SQL query.
4. The backend executes the query and returns the formatted data to the user.

*(Note: Full technical documentation is available in the attached PDF).*
