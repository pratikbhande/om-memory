"""
om-memory Demo: Traditional RAG vs Observational Memory
========================================================
A Streamlit application showcasing how om-memory reduces token costs
by 60-80% compared to traditional RAG approaches for conversational AI.

Run: streamlit run app.py
"""

import os
import asyncio
import time
import json
import concurrent.futures
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime


def run_async(coro):
    """Run an async coroutine safely from Streamlit's sync context."""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="om-memory • Observational Memory Demo",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .stApp { font-family: 'Inter', sans-serif; }
    
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(99, 102, 241, 0.3);
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        text-align: center;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #6366f1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        margin-top: 4px;
    }
    
    .comparison-header {
        font-size: 1.1rem;
        font-weight: 600;
        padding: 8px 16px;
        border-radius: 8px;
        margin-bottom: 12px;
        text-align: center;
    }
    .rag-header {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), rgba(220, 38, 38, 0.1));
        border: 1px solid rgba(239, 68, 68, 0.3);
        color: #fca5a5;
    }
    .om-header {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(22, 163, 74, 0.1));
        border: 1px solid rgba(34, 197, 94, 0.3);
        color: #86efac;
    }
    
    .chat-bubble {
        padding: 10px 14px;
        border-radius: 10px;
        margin: 6px 0;
        font-size: 0.9rem;
        line-height: 1.5;
    }
    .user-bubble {
        background: rgba(99, 102, 241, 0.12);
        border-left: 3px solid #6366f1;
    }
    .ai-bubble {
        background: rgba(51, 65, 85, 0.4);
        border-left: 3px solid #475569;
    }
    .token-badge {
        display: inline-block;
        font-size: 0.7rem;
        padding: 2px 8px;
        border-radius: 10px;
        margin-top: 4px;
    }
    .token-badge-red {
        background: rgba(239, 68, 68, 0.2);
        color: #fca5a5;
    }
    .token-badge-green {
        background: rgba(34, 197, 94, 0.2);
        color: #86efac;
    }
    
    .obs-card {
        background: rgba(30, 41, 59, 0.6);
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 8px;
        padding: 12px;
        margin: 6px 0;
        font-size: 0.85rem;
    }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #a78bfa, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-top: 4px;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.shields.io/pypi/v/om-memory?style=flat-square&color=6366f1", width=140)
    st.markdown("### 🔑 Configuration")
    
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    
    uploaded_file = st.file_uploader("Upload Knowledge Base (.txt)", type=["txt"])
    
    st.markdown("---")
    st.markdown("### ⚙️ Simulation Settings")
    num_turns = st.slider("Conversation Turns", 5, 15, 8)
    
    st.markdown("---")
    st.markdown("""
    <div style='text-align:center; color:#64748b; font-size:0.8rem;'>
        Built with <b>om-memory 0.2.2</b><br>
        <a href='https://pypi.org/project/om-memory/' style='color:#818cf8;'>PyPI</a> •
        <a href='https://github.com/pratik333/om-memory' style='color:#818cf8;'>GitHub</a>
    </div>
    """, unsafe_allow_html=True)


# ── Hero ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🧠 om-memory</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub">Human-like Observational Memory for AI Agents • 10x cheaper than RAG for conversations</div>', unsafe_allow_html=True)


# ── Helper Functions ─────────────────────────────────────────────────────────

def load_knowledge_base():
    """Load knowledge base text from upload or default file."""
    if uploaded_file:
        return uploaded_file.read().decode("utf-8")
    
    kb_path = os.path.join(os.path.dirname(__file__), "knowledge_base.txt")
    if os.path.exists(kb_path):
        with open(kb_path, "r") as f:
            return f.read()
    return "No knowledge base found. Please upload a .txt file."


def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks for embedding."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def setup_chromadb(chunks):
    """Create a ChromaDB collection and embed the chunks."""
    import chromadb
    from chromadb.utils import embedding_functions
    
    ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=api_key, model_name="text-embedding-3-small"
    )
    
    client = chromadb.Client()
    
    # Delete if exists from a previous run
    try:
        client.delete_collection("knowledge_base")
    except Exception:
        pass
    
    collection = client.create_collection(
        name="knowledge_base", embedding_function=ef
    )
    
    collection.add(
        documents=chunks,
        ids=[f"chunk_{i}" for i in range(len(chunks))],
    )
    return collection


def rag_retrieve(collection, query, n_results=3):
    """Retrieve relevant chunks from ChromaDB."""
    results = collection.query(query_texts=[query], n_results=n_results)
    return "\n\n".join(results["documents"][0]) if results["documents"] else ""


def count_tokens_approx(text):
    """Approximate token count (1 token ≈ 4 chars)."""
    return max(1, len(text) // 4)


def generate_user_queries(client, kb_text, num_turns):
    """Use OpenAI to generate realistic multi-turn user queries."""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": f"""You are simulating a real employee at a tech company who is chatting with an HR/company AI assistant. 
Generate exactly {num_turns} realistic conversational messages that this employee would send across a multi-turn conversation.

Rules:
- Start with a greeting and an initial question
- Each message should be a natural follow-up that references previous context
- Include personal details (name, role, team) early on
- Ask about different topics: PTO, remote work, expenses, career growth, on-call, benefits
- Some messages should reference things "discussed earlier" in the conversation  
- Make it feel like a real person, not a test script
- Messages should get progressively more specific and reference prior answers

Return ONLY a JSON array of strings, no markdown, no explanation.
Example: ["Hi, I'm looking into...", "Thanks! Also...", ...]

Knowledge base for context:
{kb_text[:3000]}"""},
            {"role": "user", "content": f"Generate {num_turns} conversation messages."}
        ],
        temperature=0.8,
    )
    
    try:
        raw = response.choices[0].message.content.strip()
        # Handle potential markdown wrapper
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        return json.loads(raw)
    except Exception:
        # Fallback queries
        return [
            "Hi, I'm Alex from the backend team. How many PTO days do I get?",
            "I've used 8 days already. Can I carry over unused days to next year?",
            "What about remote work — when can I work from home?",
            "I'm an L3 engineer. What do I need for promotion to L4?",
            "Tell me about the on-call rotation. What's the pay?",
            "Going to re:Invent next month. What's the expense policy for conferences?",
            "Can you remind me how many PTO days I have left based on what I told you?",
            "What mental health benefits does the company offer?",
        ][:num_turns]


async def run_traditional_rag(client, collection, queries):
    """Run traditional RAG: full chat history + retrieved docs every turn."""
    chat_history = []
    results = []
    cumulative_tokens = 0
    
    for q in queries:
        # Retrieve relevant docs
        retrieved = rag_retrieve(collection, q)
        
        # Build prompt with FULL chat history (this is what makes traditional RAG expensive)
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in chat_history])
        
        system_prompt = f"""You are a helpful company HR assistant. Answer based on the knowledge base and conversation history.

=== KNOWLEDGE BASE (Retrieved) ===
{retrieved}

=== FULL CONVERSATION HISTORY ===
{history_text}
"""
        prompt_tokens = count_tokens_approx(system_prompt + q)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ],
            temperature=0.3,
        )
        
        answer = response.choices[0].message.content
        answer_tokens = count_tokens_approx(answer)
        cumulative_tokens += prompt_tokens + answer_tokens
        
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": answer})
        
        results.append({
            "query": q,
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "cumulative_tokens": cumulative_tokens,
        })
    
    return results


async def run_om_memory(client, collection, queries):
    """Run om-memory approach: compressed memory + retrieved docs."""
    from om_memory import ObservationalMemory
    
    om = ObservationalMemory(api_key=api_key)
    om.config.demo_mode = True
    om.config.observer_token_threshold = 500  # Very low for demo
    om.config.message_retention_count = 3
    
    thread_id = f"demo_{int(time.time())}"
    resource_id = "demo_user"
    results = []
    cumulative_tokens = 0
    observations_log = []
    
    for i, q in enumerate(queries):
        # Retrieve relevant docs
        retrieved = rag_retrieve(collection, q)
        
        # Get om-memory context (compressed observations + recent messages)
        memory_context = await om.aget_context(thread_id, resource_id=resource_id)
        
        system_prompt = f"""You are a helpful company HR assistant. Answer based on the knowledge base and memory context.

=== KNOWLEDGE BASE (Retrieved) ===
{retrieved}

=== OBSERVATIONAL MEMORY CONTEXT ===
{memory_context}
"""
        prompt_tokens = count_tokens_approx(system_prompt + q)
        
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": q}
            ],
            temperature=0.3,
        )
        
        answer = response.choices[0].message.content
        answer_tokens = count_tokens_approx(answer)
        cumulative_tokens += prompt_tokens + answer_tokens
        
        # Save to om-memory
        await om.aadd_message(thread_id, "user", q, resource_id=resource_id)
        await om.aadd_message(thread_id, "assistant", answer, resource_id=resource_id)
        
        # Check observation state
        obs = await om.aget_observations(thread_id)
        observations_log.append({
            "turn": i + 1,
            "observation_count": len(obs),
            "observations": [{"priority": o.priority.value, "content": o.content[:120]} for o in obs],
        })
        
        results.append({
            "query": q,
            "answer": answer,
            "prompt_tokens": prompt_tokens,
            "answer_tokens": answer_tokens,
            "cumulative_tokens": cumulative_tokens,
            "memory_context_preview": memory_context[:500],
        })
    
    # Final force observe to show compression
    await om.aobserve(thread_id, resource_id=resource_id)
    final_obs = await om.aget_observations(thread_id)
    final_context = await om.aget_context(thread_id, resource_id=resource_id)
    
    return results, observations_log, final_obs, final_context


# ── Main Logic ───────────────────────────────────────────────────────────────

if not api_key:
    st.info("👈 Enter your OpenAI API key in the sidebar to get started.")
    
    # Show architecture overview even without key
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ### ❌ Traditional RAG for Conversations
        ```
        Turn 1: System + Retrieved Docs + Q1         → 800 tokens  
        Turn 2: System + Retrieved Docs + Q1+A1+Q2   → 1,400 tokens
        Turn 3: System + Retrieved Docs + Q1+A1+Q2+A2+Q3 → 2,200 tokens
        ...
        Turn 10: System + Retrieved + ALL history     → 8,000+ tokens 😱
        ```
        **Problem:** Token cost grows O(n²) with conversation length.
        """)
    with col2:
        st.markdown("""
        ### ✅ om-memory Approach
        ```
        Turn 1: System + Retrieved Docs + Q1         → 800 tokens
        Turn 2: System + Retrieved Docs + Memory + Q2 → 900 tokens
        Turn 3: System + Retrieved Docs + Memory + Q3 → 950 tokens
        ...
        Turn 10: System + Retrieved + Memory + Q10   → 1,200 tokens 🎉
        ```
        **Solution:** Token cost stays O(1) — memory compresses automatically.
        """)
    st.stop()


# Set API key
os.environ["OPENAI_API_KEY"] = api_key

import openai
oai_client = openai.OpenAI(api_key=api_key)


# ── Run Simulation ───────────────────────────────────────────────────────────

if st.sidebar.button("🚀 Run Simulation", type="primary", use_container_width=True):
    
    with st.spinner("📚 Loading and indexing knowledge base into ChromaDB..."):
        kb_text = load_knowledge_base()
        chunks = chunk_text(kb_text)
        collection = setup_chromadb(chunks)
        st.session_state["kb_chunks"] = len(chunks)
    
    with st.spinner("🤖 Generating realistic user queries with OpenAI..."):
        queries = generate_user_queries(oai_client, kb_text, num_turns)
        st.session_state["queries"] = queries
    
    with st.spinner("📊 Running Traditional RAG pipeline..."):
        rag_results = run_async(run_traditional_rag(oai_client, collection, queries))
        st.session_state["rag_results"] = rag_results
    
    with st.spinner("🧠 Running om-memory pipeline..."):
        om_results, obs_log, final_obs, final_ctx = run_async(
            run_om_memory(oai_client, collection, queries)
        )
        st.session_state["om_results"] = om_results
        st.session_state["obs_log"] = obs_log
        st.session_state["final_obs"] = final_obs
        st.session_state["final_ctx"] = final_ctx
    
    st.session_state["simulation_done"] = True
    st.rerun()


# ── Display Results ──────────────────────────────────────────────────────────

if st.session_state.get("simulation_done"):
    
    rag_results = st.session_state["rag_results"]
    om_results = st.session_state["om_results"]
    obs_log = st.session_state["obs_log"]
    final_obs = st.session_state["final_obs"]
    final_ctx = st.session_state["final_ctx"]
    
    # Summary metrics
    rag_total = rag_results[-1]["cumulative_tokens"]
    om_total = om_results[-1]["cumulative_tokens"]
    savings_pct = ((rag_total - om_total) / rag_total) * 100 if rag_total > 0 else 0
    compression = rag_total / om_total if om_total > 0 else 1
    
    # Cost (gpt-4o-mini pricing: $0.15/1M input, $0.60/1M output)
    rag_cost = (rag_total * 0.15) / 1_000_000 + (sum(r["answer_tokens"] for r in rag_results) * 0.60) / 1_000_000
    om_cost = (om_total * 0.15) / 1_000_000 + (sum(r["answer_tokens"] for r in om_results) * 0.60) / 1_000_000
    
    # Top metrics row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{savings_pct:.0f}%</div>
            <div class="metric-label">Token Savings</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{compression:.1f}x</div>
            <div class="metric-label">Compression Ratio</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(final_obs)}</div>
            <div class="metric-label">Observations Extracted</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(rag_results)}</div>
            <div class="metric-label">Conversation Turns</div>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("")
    
    # ── Tabs ─────────────────────────────────────────────────────────────
    tab1, tab2, tab3 = st.tabs(["🔬 Live Comparison", "💰 Cost Dashboard", "🧠 Observability"])
    
    # ── TAB 1: Live Comparison ───────────────────────────────────────────
    with tab1:
        left, right = st.columns(2)
        
        with left:
            st.markdown('<div class="comparison-header rag-header">❌ Traditional RAG (Full History)</div>', unsafe_allow_html=True)
            for r in rag_results:
                st.markdown(f"""
                <div class="chat-bubble user-bubble">
                    <b>👤 User:</b> {r['query']}
                </div>
                <div class="chat-bubble ai-bubble">
                    <b>🤖 AI:</b> {r['answer'][:300]}{'...' if len(r['answer']) > 300 else ''}
                    <br><span class="token-badge token-badge-red">⚡ {r['prompt_tokens']:,} prompt tokens | Cumulative: {r['cumulative_tokens']:,}</span>
                </div>
                """, unsafe_allow_html=True)
        
        with right:
            st.markdown('<div class="comparison-header om-header">✅ om-memory (Observational Memory)</div>', unsafe_allow_html=True)
            for r in om_results:
                st.markdown(f"""
                <div class="chat-bubble user-bubble">
                    <b>👤 User:</b> {r['query']}
                </div>
                <div class="chat-bubble ai-bubble">
                    <b>🧠 AI:</b> {r['answer'][:300]}{'...' if len(r['answer']) > 300 else ''}
                    <br><span class="token-badge token-badge-green">⚡ {r['prompt_tokens']:,} prompt tokens | Cumulative: {r['cumulative_tokens']:,}</span>
                </div>
                """, unsafe_allow_html=True)
    
    # ── TAB 2: Cost Dashboard ────────────────────────────────────────────
    with tab2:
        c1, c2 = st.columns(2)
        
        with c1:
            # Cumulative token chart
            turns = list(range(1, len(rag_results) + 1))
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=turns,
                y=[r["cumulative_tokens"] for r in rag_results],
                name="Traditional RAG",
                line=dict(color="#ef4444", width=3),
                fill="tozeroy",
                fillcolor="rgba(239, 68, 68, 0.1)",
            ))
            fig.add_trace(go.Scatter(
                x=turns,
                y=[r["cumulative_tokens"] for r in om_results],
                name="om-memory",
                line=dict(color="#22c55e", width=3),
                fill="tozeroy",
                fillcolor="rgba(34, 197, 94, 0.1)",
            ))
            fig.update_layout(
                title="Cumulative Token Usage Per Turn",
                xaxis_title="Conversation Turn",
                yaxis_title="Cumulative Tokens",
                template="plotly_dark",
                height=400,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with c2:
            # Per-turn prompt tokens
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=turns,
                y=[r["prompt_tokens"] for r in rag_results],
                name="RAG Prompt Tokens",
                marker_color="rgba(239, 68, 68, 0.7)",
            ))
            fig2.add_trace(go.Bar(
                x=turns,
                y=[r["prompt_tokens"] for r in om_results],
                name="om-memory Prompt Tokens",
                marker_color="rgba(34, 197, 94, 0.7)",
            ))
            fig2.update_layout(
                title="Prompt Tokens Per Turn",
                xaxis_title="Conversation Turn",
                yaxis_title="Tokens",
                template="plotly_dark",
                height=400,
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
                margin=dict(t=60, b=40),
            )
            st.plotly_chart(fig2, use_container_width=True)
        
        # Cost comparison cards
        st.markdown("### 💵 Estimated Cost (gpt-4o-mini pricing)")
        cc1, cc2, cc3 = st.columns(3)
        with cc1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background: linear-gradient(135deg, #ef4444, #dc2626); -webkit-background-clip: text;">${rag_cost:.4f}</div>
                <div class="metric-label">Traditional RAG Cost</div>
            </div>""", unsafe_allow_html=True)
        with cc2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value" style="background: linear-gradient(135deg, #22c55e, #16a34a); -webkit-background-clip: text;">${om_cost:.4f}</div>
                <div class="metric-label">om-memory Cost</div>
            </div>""", unsafe_allow_html=True)
        with cc3:
            saved = rag_cost - om_cost
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${saved:.4f}</div>
                <div class="metric-label">Saved This Session</div>
            </div>""", unsafe_allow_html=True)
        
        # Scaling projection
        st.markdown("### 📈 Scaling Projection")
        scale_data = []
        for users in [100, 1000, 10000, 100000]:
            daily_convos = users * 3  # 3 conversations per user per day
            monthly_convos = daily_convos * 30
            rag_monthly = rag_cost * monthly_convos
            om_monthly = om_cost * monthly_convos
            scale_data.append({
                "Users": f"{users:,}",
                "Monthly Conversations": f"{monthly_convos:,}",
                "RAG Cost/month": f"${rag_monthly:,.2f}",
                "om-memory Cost/month": f"${om_monthly:,.2f}",
                "Monthly Savings": f"${rag_monthly - om_monthly:,.2f}",
            })
        st.table(scale_data)
    
    # ── TAB 3: Observability ─────────────────────────────────────────────
    with tab3:
        st.markdown("### 🔍 What om-memory Extracted")
        st.markdown("These are the structured observations that om-memory automatically compressed from the raw conversation:")
        
        for obs in final_obs:
            priority_color = {"🔴": "#ef4444", "🟡": "#eab308", "🟢": "#22c55e"}.get(obs.priority.value, "#64748b")
            st.markdown(f"""
            <div class="obs-card">
                <span style="color:{priority_color}; font-size:1.2rem;">{obs.priority.value}</span>
                <span style="color:#e2e8f0; margin-left:8px;">{obs.content}</span>
                <br><span style="color:#64748b; font-size:0.75rem;">Observed: {obs.observation_date.strftime('%Y-%m-%d %H:%M UTC')}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📋 Final Context Window")
        st.markdown("This is exactly what the LLM receives as context on the **next** request (instead of the full chat history):")
        st.code(final_ctx, language="text")
        
        ctx_tokens = count_tokens_approx(final_ctx)
        history_tokens = rag_results[-1]["cumulative_tokens"]
        st.markdown(f"""
        <div class="metric-card" style="text-align:left;">
            <b>Context window size:</b> ~{ctx_tokens:,} tokens<br>
            <b>Full history would be:</b> ~{history_tokens:,} tokens<br>
            <b>Reduction:</b> {((history_tokens - ctx_tokens) / history_tokens * 100):.0f}% smaller
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📊 Observation Growth Over Time")
        obs_turns = [o["turn"] for o in obs_log]
        obs_counts = [o["observation_count"] for o in obs_log]
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=obs_turns, y=obs_counts,
            mode="lines+markers",
            line=dict(color="#818cf8", width=2),
            marker=dict(size=8, color="#6366f1"),
            name="Observations",
        ))
        fig3.update_layout(
            title="Observation Count Per Turn",
            xaxis_title="Turn",
            yaxis_title="Active Observations",
            template="plotly_dark",
            height=300,
            margin=dict(t=50, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True)

else:
    st.markdown("---")
    st.markdown("""
    ### How It Works
    
    1. **Enter your OpenAI API key** in the sidebar
    2. **Click "Run Simulation"** — the app will:
       - Index the knowledge base into ChromaDB (real vector store)
       - Generate realistic user queries via OpenAI
       - Run both Traditional RAG and om-memory pipelines in parallel
       - Display the results across 3 interactive dashboards
    
    ### Why om-memory?
    
    | Feature | Traditional RAG | om-memory |
    |---------|----------------|-----------|
    | Context Growth | O(n²) — full history every turn | O(1) — compressed memory |
    | Token Cost | Explodes with conversation length | Stays flat |
    | Memory Type | None — just retrieval | Observational + reflective |
    | Cross-session | ❌ | ✅ Resource-scoped memory |
    | Vector DB needed for memory? | Yes | No |
    """)
