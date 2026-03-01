"""
50-Turn Benchmark: Traditional RAG vs om-memory
Generates publication-quality graphs for LinkedIn.

Outputs:
  - demo/images/token_growth.png     (per-turn prompt tokens line chart)
  - demo/images/cumulative_cost.png  (cumulative token bar chart)
  - demo/images/savings_summary.png  (pie/summary chart with key metrics)

Usage:
  cd om-memory
  source .env   # needs OPENAI_API_KEY
  python demo/generate_graphs.py
"""
import os, sys, asyncio, time, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from pathlib import Path

# Add parent dir to path so we can import om_memory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from openai import AsyncOpenAI
from om_memory import ObservationalMemory, OMConfig

# ── Knowledge Base ──
KB = """ACME TECHNOLOGIES EMPLOYEE HANDBOOK (ABRIDGED)

PTO: 20 days/yr (1.67/mo), day-1 accrual. 5+yr tenure: 25 days. Carry-over max 5 days. 2-week advance notice for >3 days. Q4: max 3 consecutive days unless VP-approved.

REMOTE WORK: Hybrid Tue/Thu remote for all. L5+ up to 3 days/wk with manager approval. Core hours 10AM-4PM EST. VPN required always. $1,500/yr home office stipend (no rollover). Daily standup 10:15AM EST mandatory. On-site for quarterly planning weeks.

TRAVEL & EXPENSES: Meals $50/day domestic, $75/day international. Hotels $200/night domestic, $300/night international. Submit within 30 days via Expensify. Receipts required >$25. Economy class domestic; business class for international flights >6 hours. Mileage reimbursement $0.67/mile. Corporate Amex preferred; personal cards accepted with manager pre-approval.

BENEFITS: 401k match up to 6%. Health insurance (Aetna PPO or Kaiser HMO). Dental + vision included. Life insurance 2x salary. EAP available 24/7. Gym reimbursement up to $75/mo.

PROFESSIONAL DEVELOPMENT: $3,000/yr learning budget. Conference attendance requires manager approval 30 days in advance. Certifications reimbursed 100% if passed."""

# ── 50 Conversation Turns ──
QUERIES = [
    "Hi, I'm Alex Chen. I'm an L5 senior engineer on the Platform team. Been here 6 years.",
    "How many PTO days do I get with my tenure?",
    "Can I work remotely 3 days a week as an L5?",
    "What's the home office stipend amount?",
    "I want to attend KubeCon in Paris next April. What do I need?",
    "How much is the learning budget for conferences?",
    "The flight to Paris is about 8 hours. Can I fly business class?",
    "What's the international hotel allowance?",
    "What about meal expenses in Paris?",
    "Do I need receipts for a €20 coffee?",
    "When do I need to submit expense reports after the trip?",
    "Can I use my personal credit card or do I need corporate Amex?",
    "I also want to take a week of vacation after the conference. How do I request it?",
    "How many carry-over days can I bring from last year?",
    "What's the mileage rate if I drive to the airport?",
    "Is VPN required when working from the hotel in Paris?",
    "What are the core hours I need to be available during remote work?",
    "Can I skip the daily standup while at the conference?",
    "My manager wants to know about Q4 travel restrictions. What are they?",
    "If I get VP approval, can I take more than 3 days in Q4?",
    "Actually, I'm also thinking about the Tokyo tech summit in June.",
    "How long is the flight to Tokyo? About 14 hours from NYC.",
    "Would that qualify for business class too?",
    "What's the hotel limit in Tokyo?",
    "The summit registration is $2,500. Is that covered by learning budget?",
    "Can I combine the Tokyo trip with vacation days?",
    "My colleague mentioned the gym reimbursement. How much is it?",
    "What health insurance options do we have?",
    "What's the 401k match percentage?",
    "Is there life insurance included?",
    "What's the EAP program?",
    "Back to travel planning — summarize my Paris trip plan so far.",
    "What about my Tokyo trip plan?",
    "I want to get AWS Solutions Architect certified. Is that reimbursed?",
    "Do I need manager approval for certifications?",
    "How far in advance do I need to request conference attendance?",
    "My team wants to do a team dinner in NYC. What's the meal allowance?",
    "Do we need receipts for the team dinner?",
    "What if the dinner costs $350 for 5 people?",
    "Can I expense Uber rides to the restaurant?",
    "Let me now plan my Q1 calendar. I have Paris in April, what else?",
    "When is the quarterly planning week? I need to be on-site.",
    "Can I schedule Tokyo for June if I already took April vacation?",
    "How many total PTO days would both trips use?",
    "Do I have enough PTO for both trips plus some buffer?",
    "What happens if I run out of PTO?",
    "Can I request unpaid leave if needed?",
    "Let's finalize — give me a complete summary of both trip plans.",
    "What's the total estimated cost for both trips combined?",
    "One last thing — remind me of all my benefits as an L5 with 6 years tenure.",
]

# ── Plotting Helpers ──
def setup_plot_style():
    """Set up premium dark theme for all plots."""
    plt.rcParams.update({
        'figure.facecolor': '#0d1117',
        'axes.facecolor': '#161b22',
        'axes.edgecolor': '#30363d',
        'axes.labelcolor': '#e6edf3',
        'text.color': '#e6edf3',
        'xtick.color': '#8b949e',
        'ytick.color': '#8b949e',
        'grid.color': '#21262d',
        'grid.alpha': 0.6,
        'font.family': 'sans-serif',
        'font.size': 13,
        'axes.titlesize': 18,
        'axes.labelsize': 14,
        'legend.fontsize': 12,
        'figure.titlesize': 22,
    })

def save_token_growth_chart(rag_turns, om_turns, output_path):
    """Line chart: per-turn prompt tokens (RAG grows, OM stays flat)."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    turns = range(1, len(rag_turns) + 1)
    
    # RAG line — red, growing
    ax.plot(turns, rag_turns, color='#f85149', linewidth=2.5, label='Traditional RAG',
            marker='o', markersize=3, alpha=0.9)
    ax.fill_between(turns, rag_turns, alpha=0.15, color='#f85149')
    
    # OM line — green, flat
    ax.plot(turns, om_turns, color='#3fb950', linewidth=2.5, label='om-memory',
            marker='s', markersize=3, alpha=0.9)
    ax.fill_between(turns, om_turns, alpha=0.15, color='#3fb950')
    
    # Annotations
    last_rag = rag_turns[-1]
    last_om = om_turns[-1]
    saving_pct = ((last_rag - last_om) / last_rag) * 100
    
    ax.annotate(f'{last_rag:,} tokens', xy=(len(rag_turns), last_rag),
                xytext=(10, 10), textcoords='offset points',
                color='#f85149', fontsize=12, fontweight='bold')
    ax.annotate(f'{last_om:,} tokens', xy=(len(om_turns), last_om),
                xytext=(10, -20), textcoords='offset points',
                color='#3fb950', fontsize=12, fontweight='bold')
    
    ax.set_xlabel('Conversation Turn', fontweight='bold')
    ax.set_ylabel('Prompt Tokens per Turn', fontweight='bold')
    ax.set_title(f'Token Usage per Turn — RAG vs om-memory (50 turns)\n'
                 f'om-memory saves {saving_pct:.0f}% per turn by turn 50',
                 fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.8, edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(rag_turns))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved: {output_path}")

def save_cumulative_chart(rag_total, om_frontend, om_background, output_path):
    """Bar chart: total token usage breakdown."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(10, 7))
    
    categories = ['Traditional\nRAG', 'om-memory\n(total)']
    rag_vals = [rag_total, 0]
    om_front_vals = [0, om_frontend]
    om_bg_vals = [0, om_background]
    om_total = om_frontend + om_background
    
    x = np.arange(len(categories))
    width = 0.5
    
    # RAG bar
    bars1 = ax.bar(x[0], rag_total, width, color='#f85149', alpha=0.85, edgecolor='#da3633', linewidth=1.5)
    # OM stacked bars
    bars2 = ax.bar(x[1], om_frontend, width, color='#3fb950', alpha=0.85, edgecolor='#238636', linewidth=1.5, label='Front-end tokens')
    bars3 = ax.bar(x[1], om_background, width, bottom=om_frontend, color='#1f6feb', alpha=0.85, edgecolor='#1158c7', linewidth=1.5, label='Background (observer)')
    
    # Value labels
    ax.text(x[0], rag_total + 200, f'{rag_total:,}', ha='center', fontsize=14, fontweight='bold', color='#f85149')
    ax.text(x[1], om_total + 200, f'{om_total:,}', ha='center', fontsize=14, fontweight='bold', color='#3fb950')
    
    savings_pct = ((rag_total - om_total) / rag_total) * 100
    
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=14)
    ax.set_ylabel('Total Tokens (50 turns)', fontweight='bold')
    ax.set_title(f'Cumulative Token Usage — 50-Turn Conversation\n'
                 f'om-memory saves {savings_pct:.0f}% total tokens',
                 fontweight='bold', pad=15)
    ax.legend(loc='upper right', framealpha=0.8, edgecolor='#30363d')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{int(x):,}'))
    ax.set_ylim(0, rag_total * 1.15)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved: {output_path}")

def save_savings_summary(rag_total, om_total, rag_turns, om_turns, accuracy, output_path):
    """Summary infographic with key metrics."""
    setup_plot_style()
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    savings_pct = ((rag_total - om_total) / rag_total) * 100
    ratio = rag_total / om_total if om_total > 0 else 0
    last_turn_saving = ((rag_turns[-1] - om_turns[-1]) / rag_turns[-1]) * 100
    
    # Cost per 1M tokens for gpt-4o-mini
    cost_per_1m_input = 0.15  # $0.15 per 1M input tokens
    rag_cost = (rag_total / 1_000_000) * cost_per_1m_input
    om_cost = (om_total / 1_000_000) * cost_per_1m_input
    cost_saving = rag_cost - om_cost
    
    metrics = [
        (f'{savings_pct:.0f}%', 'Token Savings', f'{rag_total:,} → {om_total:,} tokens', '#3fb950'),
        (f'{last_turn_saving:.0f}%', 'Per-Turn Savings\n(at Turn 50)', f'RAG: {rag_turns[-1]:,} vs OM: {om_turns[-1]:,}', '#58a6ff'),
        (f'{accuracy:.0f}%', 'Memory Accuracy', f'Facts recalled after compression', '#d2a8ff'),
    ]
    
    for ax, (value, title, subtitle, color) in zip(axes, metrics):
        ax.text(0.5, 0.55, value, transform=ax.transAxes, fontsize=52, fontweight='bold',
                color=color, ha='center', va='center')
        ax.text(0.5, 0.25, title, transform=ax.transAxes, fontsize=16, fontweight='bold',
                color='#e6edf3', ha='center', va='center')
        ax.text(0.5, 0.1, subtitle, transform=ax.transAxes, fontsize=10,
                color='#8b949e', ha='center', va='center')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#30363d')
    
    fig.suptitle('om-memory: Human-like Memory for AI Agents', fontsize=20, fontweight='bold', y=0.98)
    fig.text(0.5, 0.88, f'50-turn HR chatbot benchmark · gpt-4o-mini · Real API calls',
             fontsize=12, color='#8b949e', ha='center')
    
    plt.tight_layout(rect=[0, 0, 1, 0.85])
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved: {output_path}")

def save_cost_projection(rag_turns, om_turns, output_path):
    """Projected dollar cost over conversation length."""
    setup_plot_style()
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # gpt-4o-mini pricing: $0.15 per 1M input tokens
    cost_per_token = 0.15 / 1_000_000
    
    # Cumulative tokens at each turn
    rag_cumul = np.cumsum(rag_turns) * cost_per_token * 1000  # in milli-dollars for readability
    om_cumul = np.cumsum(om_turns) * cost_per_token * 1000
    
    turns = range(1, len(rag_turns) + 1)
    
    ax.plot(turns, rag_cumul, color='#f85149', linewidth=2.5, label='Traditional RAG')
    ax.fill_between(turns, rag_cumul, alpha=0.15, color='#f85149')
    ax.plot(turns, om_cumul, color='#3fb950', linewidth=2.5, label='om-memory')
    ax.fill_between(turns, om_cumul, alpha=0.15, color='#3fb950')
    
    # Shade the savings area
    ax.fill_between(turns, om_cumul, rag_cumul, alpha=0.1, color='#3fb950', label='Savings')
    
    ax.set_xlabel('Conversation Turn', fontweight='bold')
    ax.set_ylabel('Cumulative Prompt Cost (× $0.001)', fontweight='bold')
    ax.set_title('Cumulative Cost — RAG vs om-memory\nSavings grow with every turn',
                 fontweight='bold', pad=15)
    ax.legend(loc='upper left', framealpha=0.8, edgecolor='#30363d')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(rag_turns))
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  📊 Saved: {output_path}")

# ── Main Benchmark ──
async def run_benchmark():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("❌ Set OPENAI_API_KEY to run this benchmark")
        return

    oai = AsyncOpenAI(api_key=api_key)
    img_dir = Path(__file__).resolve().parent / "images"
    img_dir.mkdir(exist_ok=True)

    print("=" * 65)
    print("  50-TURN BENCHMARK: Traditional RAG vs om-memory")
    print("=" * 65)

    # ── Traditional RAG ──
    print("\n  🔴 Running RAG (full history each turn)...")
    rag_history = []
    rag_cumulative = 0
    rag_per_turn = []

    for i, q in enumerate(QUERIES):
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in rag_history])
        sys_p = f"You are an HR assistant. Answer concisely.\nKB: {KB}\nHistory:\n{history_text}"
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
        print(f"    Turn {i+1:2d}/50: prompt={pt:5d}  cumulative={rag_cumulative:6d}")

    # ── om-memory ──
    print("\n  🟢 Running om-memory (compressed observations + rolling window)...")
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

    thread_id = f"bench50_{int(time.time())}"
    om_cumulative = 0
    om_per_turn = []

    for i, q in enumerate(QUERIES):
        memory_ctx = await om.aget_context(thread_id)
        sys_p = f"You are an HR assistant. Answer concisely.\nKB: {KB}\n{memory_ctx}"
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
        print(f"    Turn {i+1:2d}/50: prompt={pt:5d}  cumulative={om_cumulative:6d}  bg={bg:5d}  obs={obs_count:2d}")

    # ── Accuracy test ──
    print("\n  🔍 Running accuracy test (memory recall after compression)...")
    recall_questions = [
        ("What is my name?", ["Alex", "Chen"]),
        ("What level am I?", ["L5", "senior"]),
        ("What team am I on?", ["Platform"]),
        ("How long have I been at the company?", ["6"]),
        ("What conference am I planning to attend?", ["KubeCon"]),
        ("What certification am I pursuing?", ["AWS", "Solutions Architect"]),
        ("What's my gym reimbursement?", ["75"]),
        ("What's my 401k match?", ["6"]),
    ]
    
    passed = 0
    for question, keywords in recall_questions:
        memory_ctx = await om.aget_context(thread_id)
        sys_p = f"You are an HR assistant. Answer concisely.\nKB: {KB}\n{memory_ctx}"
        resp = await oai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": sys_p}, {"role": "user", "content": question}],
            max_tokens=80,
        )
        answer = resp.choices[0].message.content.lower()
        found = any(kw.lower() in answer for kw in keywords)
        if found:
            passed += 1
        status = "✅" if found else "❌"
        print(f"    {status} {question} → {resp.choices[0].message.content[:60]}...")
    
    accuracy = (passed / len(recall_questions)) * 100
    print(f"\n    📊 Memory accuracy: {passed}/{len(recall_questions)} ({accuracy:.0f}%)")

    # ── Results ──
    stats = await om.aget_stats(thread_id)
    bg_total = stats.total_input_tokens + stats.total_output_tokens
    om_total = om_cumulative + bg_total
    savings = ((rag_cumulative - om_total) / rag_cumulative) * 100

    print(f"\n{'=' * 65}")
    print(f"  RESULTS")
    print(f"{'=' * 65}")
    print(f"  RAG total:    {rag_cumulative:>8,} tokens")
    print(f"  OM front-end: {om_cumulative:>8,} tokens")
    print(f"  OM background:{bg_total:>8,} tokens")
    print(f"  OM total:     {om_total:>8,} tokens")
    print(f"  Savings:      {savings:>7.1f}%")
    print(f"  Accuracy:     {accuracy:>7.0f}%")

    # ── Save data ──
    data = {
        "rag_per_turn": rag_per_turn,
        "om_per_turn": om_per_turn,
        "rag_cumulative": rag_cumulative,
        "om_cumulative": om_cumulative,
        "om_background": bg_total,
        "om_total": om_total,
        "savings_pct": savings,
        "accuracy": accuracy,
    }
    data_path = img_dir / "benchmark_data.json"
    with open(data_path, 'w') as f:
        json.dump(data, f, indent=2)
    print(f"\n  💾 Data saved: {data_path}")

    # ── Generate graphs ──
    print("\n  📊 Generating graphs...")
    save_token_growth_chart(rag_per_turn, om_per_turn, str(img_dir / "token_growth.png"))
    save_cumulative_chart(rag_cumulative, om_cumulative, bg_total, str(img_dir / "cumulative_cost.png"))
    save_savings_summary(rag_cumulative, om_total, rag_per_turn, om_per_turn, accuracy, str(img_dir / "savings_summary.png"))
    save_cost_projection(rag_per_turn, om_per_turn, str(img_dir / "cost_projection.png"))

    print(f"\n  ✅ All graphs saved to {img_dir}/")
    await om.aclose()

if __name__ == "__main__":
    asyncio.run(run_benchmark())
