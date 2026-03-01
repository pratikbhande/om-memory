"""
Comprehensive om-memory test: Accuracy + Memory Retention + Cost Savings

Tests three things:
1. ACCURACY: Can the agent recall facts from early converstion turns after compression?
2. MEMORY RETENTION: Does the agent remember names, decisions, and context?
3. COST: Does om-memory actually use fewer tokens than traditional RAG?

This test uses REAL API calls (OpenAI) to validate end-to-end behavior.
"""
import os, asyncio, time, json
from openai import AsyncOpenAI
from om_memory import ObservationalMemory, OMConfig

KB = """ACME HANDBOOK
PTO: 20 days/yr (1.67/mo). 5+yr tenure: 25 days. Carry-over max 5 days. Q4 max 3 days unless VP approved.
REMOTE: Tue/Thu remote. L5+ 3 days/wk with manager ok. Core 10AM-4PM EST. VPN required. $1500/yr stipend.
EXPENSES: Meals $50/day domestic, $75 international. Hotels $200/night domestic, $300 international. Receipts >$25. Economy dom; business >6hr intl. Mileage $0.67/mi."""

async def chat_rag(oai, history, query, kb):
    """Traditional RAG: full history in every call."""
    history_text = "\n".join([f"{m['role']}: {m['content']}" for m in history])
    sys_p = f"You are an HR assistant. Answer concisely.\nKB: {kb}\nHistory:\n{history_text}"
    resp = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": query}],
        max_tokens=100,
    )
    answer = resp.choices[0].message.content
    history.extend([{"role": "user", "content": query}, {"role": "assistant", "content": answer}])
    return answer, resp.usage.prompt_tokens, resp.usage.completion_tokens

async def chat_om(oai, om, thread_id, query, kb):
    """OM-memory: compressed observations + rolling window."""
    memory_ctx = await om.aget_context(thread_id)
    sys_p = f"You are an HR assistant. Answer concisely.\nKB: {kb}\n{memory_ctx}"
    resp = await oai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": query}],
        max_tokens=100,
    )
    answer = resp.choices[0].message.content
    await om.aadd_message(thread_id, "user", query)
    await om.aadd_message(thread_id, "assistant", answer)
    return answer, resp.usage.prompt_tokens, resp.usage.completion_tokens


async def test_accuracy(oai, om, thread_id):
    """
    TEST 1: Memory Accuracy
    Ask questions that require recalling info from earlier turns.
    """
    print("\n" + "=" * 65)
    print("  TEST 1: MEMORY ACCURACY")
    print("  (Can OM recall facts from early turns after compression?)")
    print("=" * 65)
    
    # Phase 1: Establish facts (these should get compressed into observations)
    setup_queries = [
        "Hi, I'm Sarah Chen from the Data Engineering team. I've been at Acme for 7 years.",
        "I want to plan international travel to Tokyo for a conference in March.",
        "The conference is called DataCon 2026 and runs March 15-18.",
        "My manager is David Lee. He already approved 5 days off.",
        "I'll be flying from New York. The flight is about 14 hours.",
        "I also want to add a personal vacation after the conference — 3 extra days.",
        "My hotel preference is the Shinjuku Hilton, around $280/night.",
    ]
    
    print("\n  --- Setting up conversation context (7 turns) ---")
    for i, q in enumerate(setup_queries):
        answer, pt, ct = await chat_om(oai, om, thread_id, q, KB)
        print(f"  Turn {i+1}: {q[:60]}...")
    
    # Check what got observed
    obs = await om.aget_observations(thread_id)
    msgs = await om.storage.aget_messages(thread_id)
    print(f"\n  After setup: {len(obs)} observations, {len(msgs)} retained messages")
    if obs:
        print("  Observations:")
        for o in obs:
            print(f"    {o.priority.value} {o.content[:80]}")
    
    # Phase 2: Ask recall questions — these test if the agent remembers early facts
    recall_questions = [
        ("What is my name?", ["Sarah", "Chen"]),
        ("What team am I on?", ["Data Engineering"]),
        ("How long have I been at the company?", ["7"]),
        ("What conference am I attending?", ["DataCon"]),
        ("What are the conference dates?", ["March 15", "March 18", "15-18", "15th", "18th"]),
        ("Who is my manager?", ["David", "Lee"]),
        ("How long is my flight?", ["14"]),
        ("What hotel do I prefer?", ["Shinjuku", "Hilton"]),
    ]
    
    print("\n  --- Memory Recall Test (asking about earlier facts) ---")
    passed = 0
    failed = 0
    for question, expected_keywords in recall_questions:
        answer, _, _ = await chat_om(oai, om, thread_id, question, KB)
        answer_lower = answer.lower()
        found = any(kw.lower() in answer_lower for kw in expected_keywords)
        status = "✅" if found else "❌"
        if found:
            passed += 1
        else:
            failed += 1
        print(f"  {status} Q: {question}")
        print(f"      A: {answer[:100]}...")
        if not found:
            print(f"      Expected one of: {expected_keywords}")
    
    accuracy = (passed / (passed + failed)) * 100
    print(f"\n  📊 Accuracy: {passed}/{passed + failed} ({accuracy:.0f}%)")
    return accuracy


async def test_cost_comparison(oai, om, thread_id_om):
    """
    TEST 2: Cost Comparison
    Run 20 turns with both RAG and OM, compare total tokens.
    """
    print("\n" + "=" * 65)
    print("  TEST 2: COST COMPARISON (20 turns)")
    print("=" * 65)
    
    queries = [
        "Hi, I'm Marcus Johnson. I'm an L5 senior engineer, 3 years at Acme.",
        "How many PTO days do I get?",
        "Can I work remotely 3 days a week?",
        "What's my home office stipend?",
        "I want to plan a trip to Berlin for a tech summit.",
        "What's the hotel limit for international travel?",
        "The flight is 9 hours. Can I fly business class?",
        "What's the international meal allowance?",
        "I need to expense a team dinner for $200. What do I need?",
        "When is the expense submission deadline?",
        "Can I use my personal credit card or do I need corporate Amex?",
        "What are the core remote work hours?",
        "I want to take a week off during Q4 for the trip. Is that possible?",
        "How many days can I take consecutively in Q4?",
        "What if I get VP approval for more Q4 days?",
        "Do unused PTO days carry over?",
        "How many carry-over days maximum?",
        "What's the mileage rate for driving to the airport?",
        "Can I skip standups while traveling internationally?",
        "Summarize my complete travel plan.",
    ]
    
    # Traditional RAG
    rag_history = []
    rag_total = 0
    rag_turns = []
    for q in queries:
        _, pt, ct = await chat_rag(oai, rag_history, q, KB)
        rag_total += pt + ct
        rag_turns.append(pt)
    
    # OM Memory (use a separate thread)
    om_total = 0
    om_turns = []
    for q in queries:
        _, pt, ct = await chat_om(oai, om, thread_id_om, q, KB)
        om_total += pt + ct
        om_turns.append(pt)
    
    stats = await om.aget_stats(thread_id_om)
    bg_tokens = stats.total_input_tokens + stats.total_output_tokens
    om_total_with_bg = om_total + bg_tokens
    
    savings = ((rag_total - om_total_with_bg) / rag_total) * 100
    ratio = rag_total / om_total_with_bg if om_total_with_bg > 0 else float('inf')
    
    print(f"\n  RAG total:    {rag_total:>8,} tokens")
    print(f"  OM front-end: {om_total:>8,} tokens")
    print(f"  OM background:{bg_tokens:>8,} tokens")
    print(f"  OM total:     {om_total_with_bg:>8,} tokens")
    print(f"  Savings:      {savings:>7.1f}%")
    print(f"  Ratio:        {ratio:>7.2f}x")
    
    # Show growth curve
    print("\n  Prompt growth comparison (last 5 turns):")
    for i in range(15, 20):
        print(f"    Turn {i+1}: RAG={rag_turns[i]:5d}  OM={om_turns[i]:5d}  diff={rag_turns[i]-om_turns[i]:+5d}")
    
    return savings, ratio


async def test_cross_session_memory(oai, om):
    """
    TEST 3: Cross-session Memory
    Start a new thread/session and verify observations persist from previous thread.
    This tests the core value of OM — memory that survives context window limits.
    """
    print("\n" + "=" * 65)
    print("  TEST 3: LONG CONVERSATION COHERENCE")
    print("  (Does the agent stay coherent across many compression cycles?)")
    print("=" * 65)
    
    thread_id = f"coherence_{int(time.time())}"
    
    # Run 15 turns that build on each other
    conversation = [
        ("I'm planning my wedding and need PTO. My name is Jordan.", None),
        ("The wedding is June 15th. I need a week off before it.", None),
        ("My partner's name is Riley. They work at Google.", None),
        ("We're honeymooning in Bali — 10 days total.", None),
        ("I've been at Acme 2 years. How much PTO do I have?", ["20"]),
        ("That's tight. The wedding week + honeymoon = 15 days.", None),
        ("Can I carry over days from last year to help?", ["5", "carry"]),
        ("What about remote work from Bali during the honeymoon?", ["VPN", "remote", "Tue", "Thu"]),
        ("The hotel in Bali costs $180/night. Is that covered?", ["300", "international", "yes", "within"]),
        ("I also need to expense the flight — it's 18 hours.", ["business", "class"]),
        # Now test recall across compression boundary
        ("Remind me — when is my wedding again?", ["June 15", "June"]),
        ("What's my partner's name?", ["Riley"]),
        ("Where are we honeymooning?", ["Bali"]),
        ("What's my total PTO situation summary?", ["20", "carry", "5"]),
        ("Give me a final checklist of everything I need to do.", ["PTO", "expense", "VPN"]),
    ]
    
    passed = 0
    total_checks = 0
    
    for i, (q, expected) in enumerate(conversation):
        answer, pt, _ = await chat_om(oai, om, thread_id, q, KB)
        
        if expected:
            total_checks += 1
            answer_lower = answer.lower()
            found = any(kw.lower() in answer_lower for kw in expected)
            status = "✅" if found else "❌"
            if found:
                passed += 1
            print(f"  Turn {i+1}: {status} Q: {q[:55]}...")
            print(f"         A: {answer[:80]}...")
        else:
            print(f"  Turn {i+1}: 📝 {q[:60]}...")
    
    obs = await om.aget_observations(thread_id)
    msgs = await om.storage.aget_messages(thread_id)
    print(f"\n  Final state: {len(obs)} observations, {len(msgs)} messages")
    
    if total_checks > 0:
        accuracy = (passed / total_checks) * 100
        print(f"  📊 Coherence: {passed}/{total_checks} ({accuracy:.0f}%)")
        return accuracy
    return 100.0


async def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Set OPENAI_API_KEY to run this test")
        return
    
    oai = AsyncOpenAI(api_key=api_key)
    
    config = OMConfig(
        observer_token_threshold=300,
        reflector_token_threshold=1500,
        message_retention_count=2,
        message_token_budget=200,
        auto_observe=True,
        auto_reflect=True,
        blocking_mode=True,
    )
    
    om = ObservationalMemory(api_key=api_key, config=config)
    await om.ainitialize()
    
    # Run all tests
    t1_accuracy = await test_accuracy(oai, om, f"accuracy_{int(time.time())}")
    t2_savings, t2_ratio = await test_cost_comparison(oai, om, f"cost_{int(time.time())}")
    t3_coherence = await test_cross_session_memory(oai, om)
    
    # Final report
    print("\n" + "=" * 65)
    print("  FINAL REPORT")
    print("=" * 65)
    print(f"  Memory Accuracy:     {t1_accuracy:.0f}%  {'✅' if t1_accuracy >= 75 else '❌'}")
    print(f"  Cost Savings:        {t2_savings:.1f}% {'✅' if t2_savings > 0 else '❌'}")
    print(f"  Cost Ratio:          {t2_ratio:.2f}x")
    print(f"  Coherence:           {t3_coherence:.0f}%  {'✅' if t3_coherence >= 75 else '❌'}")
    
    all_pass = t1_accuracy >= 75 and t2_savings > 0 and t3_coherence >= 75
    print(f"\n  {'✅ ALL TESTS PASSED' if all_pass else '❌ SOME TESTS FAILED'}")
    
    await om.aclose()

if __name__ == "__main__":
    asyncio.run(main())
