from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
from typing import TypedDict, List
from src.retriever import retrieve_chunks
from src.hitl import should_escalate, escalate_to_human
from dotenv import load_dotenv
from langchain_groq import ChatGroq
import os

# Load environment variables
load_dotenv()

# Define state
class GraphState(TypedDict):
    query: str
    documents: List
    response: str
    escalated: bool

# ─── NODE 1: Retrieve ───
def retrieval_node(state: GraphState) -> GraphState:
    print("\n🔍 Retrieving relevant chunks...")
    docs = retrieve_chunks(state["query"])
    print(f"✅ Found {len(docs)} relevant chunks")
    return {**state, "documents": docs}

# ─── NODE 2: Generate ───
def generation_node(state: GraphState) -> GraphState:
    print("\n🤖 Generating response...")

    llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.1-8b-instant",
    temperature=0
)

    # Combine context
    context = "\n\n".join([doc.page_content for doc in state["documents"]])

    messages = [
        SystemMessage(content="""You are a helpful customer support assistant.
Answer ONLY from the given context.
If answer not found, say: 'I don't know based on available information'."""),
        
        HumanMessage(content=f"Context:\n{context}\n\nQuestion: {state['query']}")
    ]

    response = llm.invoke(messages)

    return {**state, "response": response.content}

# ─── NODE 3: Human Escalation ───
def hitl_node(state: GraphState) -> GraphState:
    human_response = escalate_to_human(state["query"])
    return {**state, "response": human_response, "escalated": True}

# ─── ROUTER ───
def route_response(state: GraphState) -> str:
    if should_escalate(state["response"], state["documents"]):
        return "escalate"
    return "answer"

# ─── BUILD GRAPH ───
def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("retrieve", retrieval_node)
    graph.add_node("generate", generation_node)
    graph.add_node("hitl", hitl_node)

    graph.set_entry_point("retrieve")

    graph.add_edge("retrieve", "generate")

    graph.add_conditional_edges(
        "generate",
        route_response,
        {
            "answer": END,
            "escalate": "hitl"
        }
    )

    graph.add_edge("hitl", END)

    return graph.compile()