def should_escalate(response: str, docs: list) -> bool:
    """
    Determines if query should be escalated to human.
    Returns True if escalation needed.
    """
    # Condition 1: No relevant documents found
    if not docs or len(docs) == 0:
        return True
    
    # Condition 2: LLM response shows uncertainty
    uncertainty_phrases = [
        "i don't know",
        "i am not sure",
        "i cannot answer",
        "unclear",
        "not enough information",
        "cannot determine"
    ]
    response_lower = response.lower()
    for phrase in uncertainty_phrases:
        if phrase in response_lower:
            return True
    
    return False

def escalate_to_human(query: str) -> str:
    """
    Simulates human-in-the-loop escalation.
    In production this would notify a real human agent.
    """
    print("\n🚨 HITL TRIGGERED — Escalating to human agent...")
    print(f"📩 Query forwarded: {query}")
    
    # Simulate human response (in real system, wait for human input)
    human_response = input("👤 Human Agent Response: ")
    return human_response