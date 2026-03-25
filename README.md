**You can edit functions in `graph_agent.py`, and in evaluator.py can change test_mode (LEGACY is langchain mode, GRAPH is langgraph)**  

# 🛠️ Prerequisites
Before you begin, ensure you have the following installed:

* Python 3.11 (Strict requirement) 

* Google Cloud API Key or other LLM Key
# ⚙️ Environment Setup
### 1. Virtual Environment Setup

It is highly recommended to use a virtual environment to manage dependencies.

**For macOS / Linux:**
```
# Create virtual environment
python -m venv venv

# Activate environment
source venv/bin/activate
```
**For Windows:**
```
# Create virtual environment
python -m venv venv

# Activate environment
venv\Scripts\activate
```

### 2. Install Dependencies

`pip install -r requirements.txt`

### 3. Environment Variables (.env)

Rename the file `.env_example` to `.env` in the root directory and add your API_KEY

# 📂 File Descriptions

* **data/:** Folder containing the raw PDF financial reports
* **langgraph_agent.py:** [MAIN WORKSPACE] This is where you will write your code. It contains the logic for:
  * PDF Ingestion: `initialize_vector_dbs()`

  *  Graph Nodes: `retrieve_node`, `grade_documents_node`, `generate_node`, `rewrite_node`.

  *  Legacy Agent: `run_legacy_agent` (The baseline for comparison).
* **evaluator.py:** The benchmark testing script. It runs a suite of test cases (Apple Revenue, Tesla R&D, Comparison, Traps) and uses "LLM-as-a-Judge" to score your agent (Pass/Fail).
* **config.py:** Configuration file that handles API key loading and initializes the LLM and Embedding models.


# 📝 Student Tasks
**You need to complete the TODO sections in `langgraph_agent.py`.**
* Task 1 (Legacy): Implement the run_legacy_agent Prompt Template to establish a baseline (langchain).

* Task 2 (Router): Implement the retrieve_node logic to route queries to "apple", "tesla", or "both".

* Task 3 (Grader): Implement the grade_documents_node to filter out irrelevant documents.

* Task 4 (Generator): Implement the generate_node to answer questions in English with Citations.

* Task 5 (Rewriter): Implement the rewrite_node to refine search queries when retrieval fails.

# 🚀 Execution Order

* Step1: `python build_rag.py`: Before running any agents, you must ingest the PDFs and convert them into vector embeddings. This allows you to experiment with different chunking strategies without re-running the evaluation logic every time.
* Step2: `python evaluator.py`: Once the database is ready, run the evaluator to benchmark your agent.
  
