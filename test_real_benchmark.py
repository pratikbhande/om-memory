"""
Benchmark: Traditional RAG vs om-memory token usage.

Simulates a 30-turn HR chatbot conversation comparing:
1. Traditional RAG: Full history in every call (tokens grow O(n²))
2. om-memory: Compressed observations + rolling window (tokens stay O(1))

Run: python test_real_benchmark.py
Requires: OPENAI_API_KEY environment variable
"""
import os, asyncio, time
from openai import AsyncOpenAI
from om_memory import ObservationalMemory, OMConfig

KNOWLEDGE_BASE = """ACME HANDBOOK
PTO: 20 days/yr (1.67/mo), day-1 accrual. Carry-over max 5 days. 2-wk notice >3 days. Q4 max 3 days unless VP ok. 5+yr tenure: 25 days.
REMOTE: Tue/Thu remote. L5+ 3 days w/manager ok. Core 10-4 EST. VPN req. $1500/yr home office. Standup 10:15 EST.
EXPENSES: Meals $50/day dom, $75 intl. Hotels $200/night dom, $300 intl. Expensify 30 days. Receipts >$25. Economy dom; biz >6hr intl. Mileage $0.67/mi."""

async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this benchmark")
        return
    
    oai = AsyncOpenAI(api_key=api_key)
    
    queries = [
        "Hi, I'm Alex from Engineering. I want to plan a vacation.",
        "How many PTO days do I get per year?",
        "I've been at the company 6 years — does that change anything?",
        "Can I work remotely on Tuesdays while traveling?",
        "What's the daily meal allowance for domestic travel?",
        "Do I need receipts for a $20 lunch?",
        "If I take 4 consecutive days in Q4, is that allowed?",
        "What about international hotel limits?",
        "Can I fly business class to London? It's a 7-hour flight.",
        "What's the mileage reimbursement rate?",
        "I want to take 2 weeks off in January. How do I request it?",
        "How many carry-over days can I bring from this year?",
        "My colleague mentioned the home office stipend — how much is it?",
        "Is VPN required when working from a hotel?",
        "What are the core hours I need to be available?",
        "Can I skip the daily standup if I'm on vacation?",
        "I need to submit my expense report — what's the deadline?",
        "What happens if I go over the $200/night hotel limit?",
        "Does the $1,500 stipend roll over to next year?",
        "Actually, I want to travel to Japan instead. What changes?",
        "What's the international meal allowance?",
        "How about the hotel rate in Japan?",
        "Is business class available for the 12-hour flight?",
        "Do I need special VPN settings for Japan?",
        "Should I factor in the time zone difference for standups?",
        "My manager has concerns about Q4 timing. What are the rules?",
        "Can I split the trip — 3 days Q4 and rest in January?",
        "How much total will the company cover for hotels for 2 weeks?",
        "Remind me — what's my total PTO balance again?",
        "Summarize our complete vacation plan and all the details.",
    ]
    
    # ===================== TRADITIONAL RAG =====================
    print("=" * 65)
    print("  TRADITIONAL RAG (full history appended each turn)")
    print("=" * 65)
    rag_history = []
    rag_cumulative = 0
    rag_per_turn = []
    
    for i, q in enumerate(queries):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in rag_history])
        sys_p = f"You are an HR assistant. Answer concisely.\nKB: {KNOWLEDGE_BASE}\nHistory:\n{history_text}"
        
        resp = await oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": q}],
            max_tokens=100,
        )
        answer = resp.choices[0].message.content
        pt = resp.usage.prompt_tokens
        ct = resp.usage.completion_tokens
        
        rag_cumulative += pt + ct
        rag_per_turn.append(pt)
        rag_history.extend([{"role": "user", "content": q}, {"role": "assistant", "content": answer}])
        print(f"  Turn {i+1:2d}: prompt={pt:5d}  total={rag_cumulative:6d}")

    # ===================== OM MEMORY =====================
    print()
    print("=" * 65)
    print("  OM MEMORY (compressed observations + rolling window)")
    print("=" * 65)
    
    config = OMConfig(
        observer_token_threshold=300,    # Compress often (~every 3 exchanges)
        reflector_token_threshold=1500,  # GC observations early
        message_retention_count=2,       # Keep only last 2 messages after compression
        message_token_budget=200,        # Tight budget for message block
        auto_observe=True,
        auto_reflect=True,
        blocking_mode=True,
    )
    
    om = ObservationalMemory(api_key=api_key, config=config)
    await om.ainitialize()
    
    thread_id = f"bench_{int(time.time())}"
    om_cumulative = 0
    om_per_turn = []
    
    for i, q in enumerate(queries):
        memory_ctx = await om.aget_context(thread_id)
        
        sys_p = f"You are an HR assistant. Answer concisely.\nKB: {KNOWLEDGE_BASE}\n{memory_ctx}"
        
        resp = await oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": q}],
            max_tokens=100,
        )
        answer = resp.choices[0].message.content
        pt = resp.usage.prompt_tokens
        ct = resp.usage.completion_tokens
        
        om_cumulative += pt + ct
        om_per_turn.append(pt)
        
        await om.aadd_message(thread_id, "user", q)
        await om.aadd_message(thread_id, "assistant", answer)
        
        stats = await om.aget_stats(thread_id)
        bg = stats.total_input_tokens + stats.total_output_tokens
        obs_count = len(await om.aget_observations(thread_id))
        msgs_count = len(await om.storage.aget_messages(thread_id))
        
        print(f"  Turn {i+1:2d}: prompt={pt:5d}  total={om_cumulative:6d}  bg={bg:5d}  obs={obs_count:2d}  msgs={msgs_count:2d}")

    # ===================== RESULTS =====================
    stats = await om.aget_stats(thread_id)
    bg_total = stats.total_input_tokens + stats.total_output_tokens
    om_total = om_cumulative + bg_total
    
    print()
    print("=" * 65)
    print("  RESULTS")
    print("=" * 65)
    print(f"  RAG total tokens:               {rag_cumulative:>8,}")
    print(f"  OM front-end tokens:            {om_cumulative:>8,}")
    print(f"  OM background tokens:           {bg_total:>8,}")
    print(f"  OM total (front + background):  {om_total:>8,}")
    print()
    
    if om_total > 0:
        savings = ((rag_cumulative - om_total) / rag_cumulative) * 100
        ratio = rag_cumulative / om_total
        icon = "✅" if savings > 0 else "❌"
        print(f"  {icon} Token savings: {savings:.1f}%")
        print(f"  RAG/OM ratio: {ratio:.2f}x")
    print()
    
    # Show per-turn comparison
    print("  Per-turn prompt tokens:")
    print(f"  {'Turn':>6}  {'RAG':>6}  {'OM':>6}  {'Saving':>8}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*8}")
    for i in range(len(queries)):
        r = rag_per_turn[i]
        o = om_per_turn[i]
        diff = r - o
        print(f"  {i+1:6d}  {r:6d}  {o:6d}  {diff:+8d}")
    
    await om.aclose()

if __name__ == "__main__":
    asyncio.run(main())
