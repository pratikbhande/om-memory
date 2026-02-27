# Google Colab Test Notebook for om-memory

This notebook demonstrates how to install and use the newly published `om-memory` package from PyPI.

## Setup
Install the package directly from PyPI:
```python
!pip install om-memory[all]
```

## Import and Initialize
Now we can import it just like any other library. You will need to set an OpenAI API key.

```python
import os
import asyncio
from om_memory import ObservationalMemory

# Replace with your actual key
os.environ["OPENAI_API_KEY"] = "sk-..."

# Initialize the memory engine
om = ObservationalMemory()
```

## Simulating a RAG Application

Let's build a basic application that combines a traditional RAG retrieval base with `om-memory` for context.

```python
import openai

def simple_rag_retrieve(query):
    # This is a mock RAG retrieval function. In a real app, this would query a Vector DB.
    documents = {
        "vacation": "Company policy allows 20 days of paid time off per year.",
        "remote": "Remote work is allowed on Tuesdays and Thursdays.",
        "expenses": "Meals are covered up to $50 per day during business travel."
    }
    for key, doc in documents.items():
        if key in query.lower():
            return doc
    return "No relevant company documents found."

async def run_hybrid_rag_chat(thread_id, user_message):
    print(f"\\n--- New Message: {user_message} ---")
    
    # 1. Retrieve the Long-Term Observational Memory Context
    memory_context = await om.aget_context(thread_id)
    
    # 2. Retrieve Traditional RAG knowledge
    rag_context = simple_rag_retrieve(user_message)
    
    # 3. Build the prompt
    prompt = f\"\"\"You are an AI assistant for a company.

{memory_context}

=== RETRIEVED KNOWLEDGE BASE ===
{rag_context}

Respond to the user.
\"\"\"
    
    # 4. Call the LLM (Using OpenAI directly here as the 'brain')
    client = openai.AsyncOpenAI()
    response = await client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_message}
        ]
    )
    
    ai_response = response.choices[0].message.content
    print(f"\\nAI Assistant: {ai_response}")
    
    # 5. Add to OM so it remembers the conversation
    await om.aadd_message(thread_id, "user", user_message)
    await om.aadd_message(thread_id, "assistant", ai_response)

# To run in Colab, we need to handle the asyncio event loop
import nest_asyncio
nest_asyncio.apply()

async def interactive_demo():
    thread_id = "colab_user_01"
    
    print("Session started. Type 'quit' to exit.")
    
    # Auto-compress more frequently for the demo
    om.config.observer_token_threshold = 200
    
    # Simulate a conversation
    messages = [
        "Hi, I'm planning my vacation for next month. Can you check the policy?",
        "I've already used 5 days this year. Let's plan for a trip to Hawaii.",
        "What days can I work remote before I leave?"
    ]
    
    for msg in messages:
        await run_hybrid_rag_chat(thread_id, msg)
        
    print(\"\\n--- FORCING COMPRESSION (OBSERVATION) ---\")
    await om.aobserve(thread_id)
    
    print(\"\\n--- FINAL MEMORY CONTEXT WINDOW FOR NEXT SESSION ---\")
    final_context = await om.aget_context(thread_id)
    print(final_context)

# Run it
asyncio.run(interactive_demo())
```
