import os
import json
from typing import TypedDict
from langgraph.graph import END, StateGraph
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
import time

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES


def initialize_vector_dbs():
    embeddings = get_embeddings()
    retrievers = {}
    for key in FILES.keys():
        persist_dir = os.path.join(DB_FOLDER, key)
        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            print(colored(f"❌ Error: Database for '{key}' not found!", "red"))
            print(colored(f"⚠️ Please run 'python build_rag.py' first.", "yellow"))
    return retrievers

RETRIEVERS = initialize_vector_dbs()


class AgentState(TypedDict):
    question: str
    documents: str
    generation: str
    search_count: int
    needs_rewrite: str


def retrieve_node(state: AgentState):
    print(colored("--- 🔍 RETRIEVING ---", "blue"))
    question = state["question"]

    try:
        time.sleep(2)
        llm = get_llm()
        options = list(FILES.keys()) + ["both", "none"]
        router_prompt = f"""You are a router. Classify the question to exactly one of these options: {', '.join(options)}

Rules:
- apple  → only about Apple Inc.
- tesla  → only about Tesla Inc.
- both   → compares or mentions both companies
- none   → unrelated to Apple or Tesla

CRITICAL: Output ONLY this exact JSON with no other text: {{"datasource": "apple"}} or {{"datasource": "tesla"}} or {{"datasource": "both"}} or {{"datasource": "none"}}

User Question: {question}"""

        response = llm.invoke(router_prompt)
        content = response.content.strip()

        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            content = content.split("```")[1].split("```")[0].strip()

        res_json = json.loads(content)
        target = res_json.get("datasource", "both").lower()

        if target not in options:
            print(colored(f"⚠️ Unknown target '{target}'. Defaulting to 'both'.", "yellow"))
            target = "both"

    except Exception as e:
        print(colored(f"⚠️ Router error: {e}. Defaulting to 'both'.", "yellow"))
        target = "both"

    print(colored(f"🎯 Routing to: {target}", "cyan"))

    if target == "none":
        print(colored("ℹ️  Out-of-scope question. Skipping retrieval.", "yellow"))
        return {
            "documents": "[No relevant financial data found for this query.]",
            "search_count": state["search_count"] + 1
        }

    targets_to_search = list(FILES.keys()) if target == "both" else [target]

    docs_content = ""
    for i, t in enumerate(targets_to_search):
        if t in RETRIEVERS:
            if i > 0:
                time.sleep(3)
            try:
                docs = RETRIEVERS[t].invoke(question)
                source_name = t.capitalize()
                docs_content += f"\n\n[Source: {source_name}]\n" + "\n".join([d.page_content for d in docs])
            except Exception as e:
                print(colored(f"❌ Retriever error for '{t}': {e}", "red"))
        else:
            print(colored(f"❌ Retriever for '{t}' not available.", "red"))

    # --- Truncate to avoid token limit ---
    max_chars = 4000
    if len(docs_content) > max_chars:
        docs_content = docs_content[:max_chars] + "\n...[truncated]"

    if not docs_content:
        docs_content = "[Retrieval returned no results.]"

    return {"documents": docs_content, "search_count": state["search_count"] + 1}


def grade_documents_node(state: AgentState):
    print(colored("--- ⚖️ GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]

    # --- Truncate before grading ---
    if len(documents) > 2000:
        documents = documents[:2000] + "\n...[truncated]"

    try:
        time.sleep(2)
        llm = get_llm()

        # --- Plain string prompt — works for ALL providers including Gemini ---
        prompt = f"""You are a binary relevance judge for a financial RAG system.
Output ONLY one word — nothing else:
- yes → document is relevant, proceed to generate
- no  → document is irrelevant, trigger rewrite

User question: {question}

Retrieved document:
{documents}

Your answer (yes or no):"""

        response = llm.invoke(prompt)
        content = response.content.strip().lower()
        grade = "yes" if content.startswith("yes") else "no"

    except Exception as e:
        print(colored(f"   ⚠️ Grading error: {e}. Defaulting to 'no'.", "yellow"))
        grade = "no"

    if grade == "yes":
        print(colored("   ✅ RELEVANT → Proceeding to Generate", "green"))
    else:
        print(colored("   ❌ IRRELEVANT → Triggering Rewrite", "red"))

    return {"needs_rewrite": grade}


def generate_node(state: AgentState):
    print(colored("--- ✍️ GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]

    fallback_signals = [
        "[No relevant financial data found for this query.]",
        "[Retrieval returned no results.]",
    ]
    if not documents.strip() or any(sig in documents for sig in fallback_signals):
        print(colored("   ⚠️ No usable context. Returning 'I don't know'.", "yellow"))
        return {"generation": "I don't know. No relevant financial data was found to answer this question."}

    # --- Truncate before generating ---
    if len(documents) > 3000:
        documents = documents[:3000] + "\n...[truncated]"

    try:
        time.sleep(2)
        llm = get_llm()

        # --- Plain string prompt — works for ALL providers including Gemini ---
        prompt = f"""You are a strict financial analyst assistant.
Answer using ONLY the provided context. Do not use outside knowledge.

Rules:
1. ANNUAL DATA PRIORITY: Always prefer full-year/twelve-month figures over quarterly ones.
2. CITATIONS: Every factual claim MUST cite its source (e.g., [Source: Apple], [Source: Tesla]).
3. HONESTY: If the context lacks the answer, respond exactly: "I don't know."
4. NO HALLUCINATION: Never fabricate numbers or dates.

Context:
{documents}

Question: {question}

Answer:"""

        response = llm.invoke(prompt)
        answer = response.content.strip()

        if not answer:
            return {"generation": "I don't know. The model did not return an answer."}

        print(colored("   ✅ Answer generated successfully.", "green"))
        return {"generation": answer}

    except Exception as e:
        print(colored(f"   ❌ Generation error: {e}", "red"))
        return {"generation": "I don't know. An error occurred during answer generation."}


def rewrite_node(state: AgentState):
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]

    try:
        time.sleep(2)
        llm = get_llm()
        prompt = f"""The previous search for '{question}' yielded irrelevant results.
Please rephrase this question to be more specific or use better keywords for a financial search engine.
Output ONLY the new question text."""

        response = llm.invoke(prompt)
        new_query = response.content.strip()
        print(f"   New Question: {new_query}")
        return {"question": new_query}

    except Exception as e:
        print(colored(f"   ⚠️ Rewrite error: {e}. Keeping original question.", "yellow"))
        return {"question": question}


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade_documents", grade_documents_node)
    workflow.add_node("generate", generate_node)
    workflow.add_node("rewrite", rewrite_node)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    def decide_to_generate(state):
        if state["needs_rewrite"] == "yes":
            return "generate"
        else:
            if state["search_count"] > 2:
                print("   (Max retries reached, generating anyway...)")
                return "generate"
            return "rewrite"

    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"generate": "generate", "rewrite": "rewrite"},
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()


def run_graph_agent(question: str):
    app = build_graph()
    inputs = {"question": question, "search_count": 0, "needs_rewrite": "no", "documents": "", "generation": ""}
    result = app.invoke(inputs)
    return result["generation"]


# --- Legacy ReAct Agent ---
def run_legacy_agent(question: str):
    print(colored("--- 🤖 RUNNING LEGACY AGENT (ReAct) ---", "magenta"))

    from langchain_core.tools.retriever import create_retriever_tool
    from langgraph.prebuilt import create_react_agent as lg_create_react_agent

    tools = []
    for key, retriever in RETRIEVERS.items():
        tools.append(create_retriever_tool(
            retriever,
            f"search_{key}_financials",
            f"Searches {key.capitalize()}'s financial data."
        ))

    if not tools:
        return "System Error: No tools available."

    llm = get_llm()

    tool_names = "\n".join([f"- {t.name}: {t.description}" for t in tools])
    tool_list = ", ".join([t.name for t in tools])

    system_prompt = f"""You are a financial analyst assistant with access to Apple and Tesla financial data.

Available tools:
{tool_names}

Use the following ReAct format STRICTLY:
Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, must be one of [{tool_list}]
Action Input: the input to the action
Observation: the result of the action
... (repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Behavioral Rules:
1. Language: The Final Answer must be in English.
2. Precision: Distinguish strictly between 2024, 2023, and 2022 data.
3. Honesty: If data is missing from the tool observations, say "I don't know." Do not guess.
4. Citations: Always cite the source tool used (e.g., [Source: Apple], [Source: Tesla]).
"""

    agent_executor = lg_create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    try:
        result = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
        final_message = result["messages"][-1]
        if hasattr(final_message, "content"):
            return final_message.content
        elif isinstance(final_message, dict):
            return final_message.get("content", str(final_message))
        else:
            return str(final_message)
    except Exception as e:
        return f"Legacy Agent Error: {e}"