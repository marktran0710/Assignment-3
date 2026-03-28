# 🤖 Assignment 3 — Financial RAG Agent with LangGraph

---

## 🛠️ Prerequisites

- Python 3.11
- API Key for your LLM provider (Google / Groq / OpenAI / Anthropic)

---

## ⚙️ Setup

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure .env
LLM_PROVIDER=google
GOOGLE_API_KEY=your_key_here
GOOGLE_MODEL=gemini-2.0-flash
```

---

## 📂 Project Structure

| File                 | Description                                     |
| -------------------- | ----------------------------------------------- |
| `data/`              | Raw PDF financial reports                       |
| `build_rag.py`       | Builds Chroma vector database from PDFs         |
| `langgraph_agent.py` | Main agent logic (nodes + graph)                |
| `evaluator.py`       | Benchmark — 14 test cases, LLM-as-Judge scoring |
| `config.py`          | LLM + embedding model factory                   |

---

## 📝 Tasks Completed

### Task A — Legacy ReAct Agent (LangChain Baseline)

- Implemented `run_legacy_agent()` using `langgraph.prebuilt.create_react_agent`
- Built dynamic tool list from RETRIEVERS
- Added ReAct format instructions + behavioral rules in system prompt
- **Fix:** `create_react_agent` moved from `langchain.agents` to `langgraph.prebuilt` in LangChain v1.x — updated import accordingly

### Task B — Router (retrieve_node)

- LLM classifies query entity → routes to `apple`, `tesla`, `both`, or `none`
- Outputs strict JSON `{"datasource": "..."}` to prevent hallucinated tool names
- **Fix:** LLM was returning `apple_financials` instead of `apple` — tightened prompt with exact JSON examples

### Task C — Grader (grade_documents_node)

- Binary judge: `yes` = relevant → generate, `no` = irrelevant → rewrite
- Uses plain string prompt (not `SystemMessage`) for cross-provider compatibility
- Truncates documents to 2000 chars before grading to avoid token limit
- **Fix:** `SystemMessage` caused `ValueError` on Gemini — replaced with single plain string prompt

### Task D — Generator (generate_node)

- Answers using ONLY retrieved context with mandatory `[Source: X]` citations
- Prioritizes annual/full-year figures over quarterly
- Returns honest `"I don't know"` when context is missing or empty
- **Fix:** `ChatPromptTemplate` with `SystemMessage` crashed on Gemini — replaced with plain f-string prompt

### Task E — Rewriter (rewrite_node)

- Rephrases failed queries with better financial keywords
- Plain string prompt for provider compatibility
- Gracefully keeps original question if rewrite fails
- **Fix:** Removed `@retry_logic` decorator from all nodes — retry loop was masking real errors and causing `RetryError` crashes

---

## 🚀 Execution Steps

### Step 1 — Build vector database

```bash
python build_rag.py
```

### Step 2 — Run LangChain baseline (LEGACY mode)

```python
# evaluator.py
TEST_MODE = "LEGACY"
```

```bash
python evaluator.py
```

### Step 3 — Run LangGraph agent (GRAPH mode)

```python
# evaluator.py
TEST_MODE = "GRAPH"
```

```bash
python evaluator.py
```

> Compare LEGACY vs GRAPH scores to evaluate LangGraph improvement

### Step 4 — Change embedding model and rebuild

```python
# config.py — change from multilingual to English-only
LOCAL_EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
```

```bash
# Rebuild DB with new embedding model
rmdir /s /q chroma_db     # Windows
rm -rf chroma_db/         # macOS/Linux
python build_rag.py
```

### Step 5 — Run LangGraph again with new embedding model

```bash
python evaluator.py        # TEST_MODE = "GRAPH"
```

> Compare GRAPH (MiniLM) vs GRAPH (mpnet) to evaluate embedding model effect

---

## 📊 Results Summary

| Mode                     | Embedding Model             | Score       |
| ------------------------ | --------------------------- | ----------- |
| LEGACY (LangChain ReAct) | MiniLM multilingual 384-dim | baseline    |
| GRAPH (LangGraph)        | MiniLM multilingual 384-dim | **9/14** ✅ |
| GRAPH (LangGraph)        | mpnet English-only 768-dim  | **5/14** ❌ |

> **Finding:** Multilingual MiniLM outperforms larger English-only mpnet for Chinese/mixed language queries against English PDFs — language alignment matters more than vector dimension size.

---

## ⚠️ Common Issues

| Error                              | Cause                                                  | Fix                                                                                                             |
| :--------------------------------- | :----------------------------------------------------- | :-------------------------------------------------------------------------------------------------------------- |
| `Quota Exceeded / RateLimit`       | API limits reached on primary provider (Google/OpenAI) | Switch `LLM_PROVIDER` to `groq` in `.env` for high-limit inference.                                             |
| `Lower Accuracy / Poor Reasoning`  | Using Groq (Llama-3/Mixtral) as fallback               | Groq is faster but may struggle with complex financial RAG logic; refine prompts for more explicit JSON output. |
| `dimension 384 vs 768 mismatch`    | Embedding model changed (MiniLM vs MPNet)              | Delete `chroma_db/` folder and run `python build_rag.py` to rebuild the vector database.                        |
| `cannot import create_react_agent` | LangChain v1.x breaking change                         | Import from `langgraph.prebuilt` instead of the legacy `langchain.agents` module.                               |
| `RetryError / Workflow Crashes`    | Token limit or rate limit hit                          | Add `time.sleep(2)` in loops and ensure prompt formatting matches the specific LLM requirements.                |
