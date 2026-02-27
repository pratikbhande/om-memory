import asyncio
import os
from om_memory import ObservationalMemory

async def main():
    # Make sure OPENAI_API_KEY is set in environment, or pass it explicitly
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set OPENAI_API_KEY environment variable.")
        return
        
    # 1. Initialize OM with default settings (SQLite storage, OpenAI provider)
    om = ObservationalMemory()
    
    # 2. Get context for a specific user thread
    thread_id = "user_demo_1"
    
    context = await om.aget_context(thread_id=thread_id)
    print("--- CONTEXT ---")
    print(context)
    print("---------------\n")
    
    # Simulate a conversation
    user_msg = "Hi, I'm planning a trip to Japan next Spring."
    ai_msg = "Japan is beautiful in Spring! Have you decided which cities to visit?"
    
    print(f"User: {user_msg}")
    print(f"AI:   {ai_msg}\n")
    
    # 3. Add messages to OM so it remembers them for next time
    await om.aadd_message(thread_id, "user", user_msg)
    await om.aadd_message(thread_id, "assistant", ai_msg)
    
    # 4. Trigger manual observation (usually auto-triggered based on token thresholds)
    print("Triggering manual observation to compress memory...")
    await om.aobserve(thread_id)
    
    # 5. Get context again to see the compressed observation
    context_after = await om.aget_context(thread_id=thread_id)
    print("\n--- NEW CONTEXT ---")
    print(context_after)
    print("-------------------")
    
if __name__ == "__main__":
    asyncio.run(main())
