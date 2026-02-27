def render_om_dashboard(om, thread_id: str = None):
    """
    Renders a Streamlit dashboard showing OM's internal state.
    
    Can be embedded in any Streamlit app:
    
        import streamlit as st
        from om_memory.observability.dashboard import render_om_dashboard
        
        render_om_dashboard(om, thread_id="user_123")
    """
    try:
        import streamlit as st
    except ImportError:
        print("Streamlit not installed. Install with `pip install streamlit` or `pip install om-memory[dashboard]`.")
        return
        
    st.title("🧠 om-memory Dashboard")
    
    if not thread_id:
        st.info("Please provide a thread_id to view specifics.")
        return
        
    st.header(f"Thread: {thread_id}")
    
    import asyncio
    
    # We create a new event loop just for pulling stats synchronously if in a sync context.
    try:
        loop = asyncio.get_running_loop()
        stats = loop.run_until_complete(om.aget_stats(thread_id))
        report = loop.run_until_complete(om.aget_savings_report(thread_id))
        observations = loop.run_until_complete(om.aget_observations(thread_id))
    except RuntimeError:
        stats = asyncio.run(om.aget_stats(thread_id))
        report = asyncio.run(om.aget_savings_report(thread_id))
        observations = asyncio.run(om.aget_observations(thread_id))
        
    # Section 1: Topline Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("OM Cost", f"${report['om_cost']:.4f}")
    col2.metric("Est. RAG Cost", f"${report['estimated_rag_cost']:.4f}")
    col3.metric("Savings ($)", f"${report['savings_dollars']:.4f}", f"{report['savings_percentage']}%")
    col4.metric("Compression", f"{report['compression_ratio']}x")
    
    # Section 2: Timeline
    st.subheader("Current Observations")
    if not observations:
        st.write("No observations recorded yet.")
    else:
        for obs in observations:
            color = "red" if obs.priority == getattr(obs.priority, "CRITICAL", "🔴") else "yellow" if obs.priority == getattr(obs.priority, "IMPORTANT", "🟡") else "green"
            st.markdown(f"**{obs.priority.value} {obs.observation_date.strftime('%Y-%m-%d %H:%M')}** - {obs.content}")
            if obs.referenced_date or obs.relative_date:
                st.caption(f"*Context: {obs.relative_date or obs.referenced_date}*")
                
    st.subheader("Detailed Stats")
    st.json(stats.model_dump())
