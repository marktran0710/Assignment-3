import os
from typing import Annotated, List, TypedDict, Literal
from langgraph.graph import END, StateGraph
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from termcolor import colored
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable

from config import get_embeddings, get_llm, DATA_FOLDER, DB_FOLDER, FILES


retry_logic = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((ResourceExhausted, ServiceUnavailable, Exception))
)


def initialize_vector_dbs():
    embeddings = get_embeddings()
    retrievers = {}
    
    # 這裡不再負責建立資料庫，只負責讀取
    for key in FILES.keys():
        persist_dir = os.path.join(DB_FOLDER, key)

        if os.path.exists(persist_dir):
            vectorstore = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
            retrievers[key] = vectorstore.as_retriever(search_kwargs={"k": 3})
        else:
            print(colored(f"❌ Error: Database for '{key}' not found!", "red"))
            print(colored(f"⚠️ Please run 'python build_rag.py' first to build the vector index.", "yellow"))
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
    
    docs_content = ""

    # --- [START] ---
    
    # Hint：You can use a prompt like below to guide the LLM
    # router_prompt = """
    # Analyze the user question and route it to the correct data source.
    # Options: "apple", "tesla", "both", "none".
    # Output only the option name in JSON format: {"datasource": "..."}
    # Question: {question}
    # """
    
    # TODO: 1. Invoke LLM with router_prompt
    # TODO: 2. Parse JSON output
    # TODO: 3. Set `target` variable based on output
    print(colored("⚠️ WARNING: You need to implement the routing logic", "yellow"))
    target = "both" # <--- Need to remove this hardcoded line after implementing the above steps and get the target from LLM output
    
    # --- [END] ---

    if target == "apple" or target == "both":
        if "apple" in RETRIEVERS:
            docs = RETRIEVERS["apple"].invoke(question)
            docs_content += "\n\n[Source: Apple 10-K]\n" + "\n".join([d.page_content for d in docs])
            
    if target == "tesla" or target == "both":
        if "tesla" in RETRIEVERS:
            docs = RETRIEVERS["tesla"].invoke(question)
            docs_content += "\n\n[Source: Tesla 10-K]\n" + "\n".join([d.page_content for d in docs])

    return {"documents": docs_content, "search_count": state["search_count"] + 1}

@retry_logic
def grade_documents_node(state: AgentState): 
    print(colored("--- ⚖️ GRADING ---", "yellow"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm()

    system_prompt = """You are a grader assessing relevance. 
    Does the retrieved document contain information related to the user question?
    
    CRITICAL: You must answer with ONLY one word: 'yes' or 'no'. Do not add any explanation."""
    
    msg = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Retrieved document context: \n\n {documents} \n\n User question: {question}")
    ]
    
    response = llm.invoke(msg)
    content = response.content.strip().lower()
    
    if "yes" in content:
        grade = "yes"
    else:
        grade = "no"
    
    print(f"   Relevance Grade: {grade} (Raw: {content})")
    return {"needs_rewrite": grade}

@retry_logic
def generate_node(state: AgentState):
    print(colored("--- ✍️ GENERATING ---", "green"))
    question = state["question"]
    documents = state["documents"]
    llm = get_llm() 
    # You can modify prompt, write the prompt LLM can generate the final answer based on the retrieved documents
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a financial analyst. Use the provided context to answer the question. \n"
                   "If the context doesn't contain the answer, say you don't know. \n"
                   "ALWAYS cite the source (e.g., [Source: Apple]).\n\nContext:\n{context}"),
        ("human", "{question}"),
    ])
    
    chain = prompt | llm
    response = chain.invoke({"context": documents, "question": question})
    return {"generation": response.content}

@retry_logic
def rewrite_node(state: AgentState): 
    print(colored("--- 🔄 REWRITING QUERY ---", "red"))
    question = state["question"]
    llm = get_llm()
    # You can modify msg, write the prompt LLM can rewrite the question to be more specific
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
    result = app.invoke(inputs)
    return result["generation"]

def run_legacy_agent(question: str):
    print(colored("--- 🤖 RUNNING LEGACY AGENT (Linear) ---", "magenta"))
    AgentExecutor = None
    create_tool_calling_agent = None
    create_retriever_tool = None
    hub = None
    try:
        from langchain.agents import AgentExecutor, create_react_agent
    except ImportError:
        try:
            from langchain.agents.agent import AgentExecutor
        except ImportError:
            pass
    try:
        from langchain.agents import create_tool_calling_agent
    except ImportError:
        pass
    try:
        from langchain.tools.retriever import create_retriever_tool
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain_core.prompts import PromptTemplate
        from langchain.tools.render import render_text_description
    except ImportError:
        try:
            from langchain.agents.agent_toolkits import create_retriever_tool
        except ImportError:
            pass
    try:
        from langchain import hub
    except ImportError:
        pass

    tools = []
    if "apple" in RETRIEVERS:
        tools.append(create_retriever_tool(
            RETRIEVERS["apple"], 
            "search_apple_financials", 
            "Searches Apple's 2024 financial statements."
        ))
    if "tesla" in RETRIEVERS:
        tools.append(create_retriever_tool(
            RETRIEVERS["tesla"], 
            "search_tesla_financials", 
            "Searches Tesla's 2024 10-K report."
        ))

    if not tools:
        return "System Error: No tools available."

    llm = get_llm()


    # ============================================================
    # TODO: Define the ReAct Prompt Template
    # ============================================================
    # Your task is to write the prompt that tells the LLM how to reason and act and must let LLM answer in English.
    # The ReAct framework REQUIRES the following structure in your string:
    #
    # 1. Description of available tools: {tools}
    # 2. Instruction on the output format:
    #    - Thought: ...
    #    - Action: ... (must be one of [{tool_names}])
    #    - Action Input: ...
    #    - Observation: ...
    # 3. The input question: {input}
    # 4. The history of thoughts/actions: {agent_scratchpad}
    #
    # Write your template string below:

    template = """
    
    """
    prompt = PromptTemplate.from_template(template)
    prompt = prompt.partial(
            tools=render_text_description(tools),
            tool_names=", ".join([t.name for t in tools])
    )

    def formatting_error_handler(error) -> str:
        error_str = str(error)
        if "Final Answer:" in error_str:
            return error_str.split("Final Answer:")[-1].strip()
        return "Agent failed to parse correctly, but here is the raw thought: " + error_str[:100]
    
    agent = create_react_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(
            agent=agent, 
            tools=tools, 
            verbose=False,
            handle_parsing_errors=formatting_error_handler,
            max_iterations=5
    )

    try:
        result = agent_executor.invoke({"input": question})
        return result["output"]
    except Exception as e:
        return f"Legacy Agent Error: {e}"