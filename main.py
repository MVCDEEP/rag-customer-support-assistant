from src.graph import build_graph, GraphState
from src.ingest import ingest_pdf
import os

def run_ingestion():
    """Run this once to process your PDF"""
    pdf_path = "data/knowledge_base.pdf"
    if not os.path.exists(pdf_path):
        print("❌ Please add a PDF file at data/knowledge_base.pdf")
        return
    ingest_pdf(pdf_path)

def run_chatbot():
    """Main chatbot loop"""
    print("\n" + "="*50)
    print("🤖 RAG Customer Support Assistant")
    print("="*50)
    print("Type 'quit' to exit\n")
    
    graph = build_graph()
    
    while True:
        query = input("👤 You: ").strip()
        
        if query.lower() == 'quit':
            print("👋 Goodbye!")
            break
            
        if not query:
            continue
        
        # Initial state
        initial_state: GraphState = {
            "query": query,
            "documents": [],
            "response": "",
            "escalated": False
        }
        
        # Run through LangGraph
        result = graph.invoke(initial_state)
        
        # Display result
        if result["escalated"]:
            print(f"\n✅ Human Agent: {result['response']}\n")
        else:
            print(f"\n🤖 Assistant: {result['response']}\n")

if __name__ == "__main__":
    print("Choose mode:")
    print("1. Ingest PDF (run first time)")
    print("2. Run Chatbot")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        run_ingestion()
    elif choice == "2":
        run_chatbot()
    else:
        print("Invalid choice")