import os
import json
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES


# Generic Retry Logic (Provider agnostic)
retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception)
)


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
            continue
    
    return retrievers

RETRIEVERS = initialize_vector_dbs()


class AgentState(TypedDict):
    question: str
    documents: str
    generation: str
    search_count: int
    needs_rewrite: str


@retry_logic
def retrieve_node(state: AgentState):
    print(colored("--- 🔍 RETRIEVING ---", "blue"))
    question = state["question"]
    llm = get_llm()

    options = list(FILES.keys()) + ["both", "none"]
    router_prompt = f"""
    Analyze the user question and route it to the correct financial data source
    based on the ENTITY mentioned in the query.

    Rules:
    - "apple"  → question mentions Apple Inc., AAPL, iPhone, Tim Cook, or Apple products
    - "tesla"  → question mentions Tesla Inc., TSLA, Elon Musk (in Tesla context), EVs by Tesla
    - "both"   → question mentions or compares BOTH companies
    - "none"   → question is completely unrelated to Apple or Tesla financials

    Options: {', '.join(options)}
    Output ONLY valid JSON: {{"datasource": "..."}}

    User Question: {question}
    """

    try:
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

    # --- Invoke retrievers based on entity ---
    docs_content = ""

    if target == "none":
        print(colored("ℹ️  Out-of-scope question. Skipping retrieval.", "yellow"))
        return {
            "documents": "[No relevant financial data found for this query.]",
            "search_count": state["search_count"] + 1
        }

    targets_to_search = list(FILES.keys()) if target == "both" else [target]

    for t in targets_to_search:
        if t in RETRIEVERS:
            docs = RETRIEVERS[t].invoke(question)
            source_name = t.capitalize()
            docs_content += f"\n\n[Source: {source_name}]\n" + "\n".join([d.page_content for d in docs])
        else:
            print(colored(f"❌ Retriever for '{t}' not available.", "red"))

    if not docs_content:
        docs_content = "[Retrieval returned no results.]"

    return {"documents": docs_content, "search_count": state["search_count"] + 1}

@retry_logic
def grade_documents_node(state: AgentState):
    print(colored("--- ⚖️ GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    system_prompt = """You are a binary relevance judge for a financial RAG system.

                        Evaluate whether the retrieved document contains information useful for answering the user's question.

                        Output ONLY one word — nothing else:
                        - yes  → document is relevant, proceed to generate an answer
                        - no   → document is irrelevant or noisy, trigger a query rewrite"""

    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"User question: {question}\n\nRetrieved document:\n{documents}")
    ]

    try:
        response = llm.invoke(msg)
        content = response.content.strip().lower()
        # Extract first word only — guards against verbose LLM responses
        grade = "yes" if content.startswith("yes") else "no"

    except Exception as e:
        print(colored(f"   ⚠️ Grading error: {e}. Defaulting to 'no'.", "yellow"))
        grade = "no"

    if grade == "yes":
        print(colored("   ✅ RELEVANT → Proceeding to Generate", "green"))
    else:
        print(colored("   ❌ IRRELEVANT → Triggering Rewrite", "red"))

    return {"needs_rewrite": grade}


@retry_logic
def generate_node(state: AgentState):
    print(colored("--- ✍️ GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    # --- Detect empty/fallback documents before invoking LLM ---
    fallback_signals = [
        "[No relevant financial data found for this query.]",
        "[Retrieval returned no results.]",
    ]
    if not documents.strip() or any(sig in documents for sig in fallback_signals):
        print(colored("   ⚠️ No usable context. Returning 'I don't know'.", "yellow"))
        return {"generation": "I don't know. No relevant financial data was found to answer this question."}

    system_prompt = """You are a strict financial analyst assistant.

                        Your job is to answer the user's question using ONLY the provided context.

                        Rules:
                        1. CITATIONS: Every factual claim MUST be followed by its source tag exactly as it appears 
                        in the context (e.g., [Source: Apple], [Source: Tesla]).
                        2. HONESTY: If the context does not contain enough information to answer the question — 
                        even partially — respond with exactly: "I don't know."
                        3. NO HALLUCINATION: Do not infer, assume, or use knowledge outside the provided context.
                        4. PARTIAL ANSWERS: If only part of the question can be answered from context, answer 
                        what you can with citations, then state "I don't know" for the missing parts.
                        5. FORMAT: Write in clear, concise prose. Do not fabricate numbers or dates.

    Context:
    {context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{question}"),
    ])

    chain = prompt | llm

    try:
        response = chain.invoke({"context": documents, "question": question})
        answer = response.content.strip()

        # --- Safety net: if LLM returns empty, return honest fallback ---
        if not answer:
            print(colored("   ⚠️ LLM returned empty response.", "yellow"))
            return {"generation": "I don't know. The model did not return an answer."}

        print(colored("   ✅ Answer generated successfully.", "green"))
        return {"generation": answer}

    except Exception as e:
        print(colored(f"   ❌ Generation error: {e}", "red"))
        return {"generation": "I don't know. An error occurred during answer generation."}

@retry_logic
def rewrite_node(state: AgentState): 
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    
    msg = [ 
        HumanMessage(content=f"The previous search for '{question}' yielded irrelevant results. \n"
                             f"Please rephrase this question to be more specific or use better keywords for a financial search engine. \n"
                             f"Output ONLY the new question text.")
    ]
    response = llm.invoke(msg)
    new_query = response.content.strip()
    print(f"   New Question: {new_query}")
    return {"question": new_query}

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
        {
            "generate": "generate",
            "rewrite": "rewrite"
        },
    )

    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)

    return workflow.compile()

def run_graph_agent(question: str):
    app = build_graph()
    inputs = {"question": question, "search_count": 0, "needs_rewrite": "no", "documents": "", "generation": ""}
    # Using stream to see progress if needed, but invoke is fine for simple return
    result = app.invoke(inputs)
    return result["generation"]

# --- Legacy ReAct Agent ---
def run_legacy_agent(question: str):
    print(colored("--- 🤖 RUNNING LEGACY AGENT (ReAct) ---", "magenta"))

    from langchain_core.tools.retriever import create_retriever_tool
    from langgraph.prebuilt import create_react_agent as lg_create_react_agent

    # --- Build tools ---
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

    # --- Build tool descriptions dynamically (replaces {tools} placeholder) ---
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
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Behavioral Rules:
1. Language: The Final Answer must be in English.
2. Precision: Distinguish strictly between 2024, 2023, and 2022 data.
3. Honesty: If data is missing from the tool observations, say "I don't know." Do not guess.
4. Citations: Always cite the source tool used (e.g., [Source: Apple], [Source: Tesla]).
"""

    # --- LangGraph native ReAct agent (replaces deprecated create_react_agent) ---
    agent_executor = lg_create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
    )

    try:
        result = agent_executor.invoke({"messages": [{"role": "user", "content": question}]})
        messages = result["messages"]
        final_message = messages[-1]
        
        # --- Always extract string content ---
        if hasattr(final_message, "content"):
            return final_message.content
        elif isinstance(final_message, dict):
            return final_message.get("content", str(final_message))
        else:
            return str(final_message)
            
    except Exception as e:
        return f"Legacy Agent Error: {e}"