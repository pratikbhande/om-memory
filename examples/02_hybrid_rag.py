import asyncio
import os
from om_memory import ObservationalMemory

async def main():
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable.")
        return
        
    # Simulated RAG system
    def rag_retrieve(query: str):
        return "Company Policy: Employees get 20 vacation days per year."
        
    om = ObservationalMemory()
    thread_id = "employee_789"
    
    user_msg = "How many vacation days do I have left? I took 5 days in March."
    
    # Get conversational memory (OM)
    memory_context = await om.aget_context(thread_id)
    
    # Get knowledge memory (RAG)
    rag_context = rag_retrieve(user_msg)
    
    # Combine both in prompt
    prompt = f"""You are an HR assistant.
    
{memory_context}

=== KNOWLEDGE BASE ===
{rag_context}

User: {user_msg}
"""
    
    print("--- FULL PROMPT SENT TO LLM ---")
    print(prompt)
    print("-------------------------------")
    
    # Simulate LLM response
    ai_msg = "Based on company policy of 20 days, and the 5 days you took in March, you have 15 days left."
    
    await om.aadd_message(thread_id, "user", user_msg)
    await om.aadd_message(thread_id, "assistant", ai_msg)

if __name__ == "__main__":
    asyncio.run(main())
