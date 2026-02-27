from typing import Dict
from om_memory.models import OMStats, OMConfig

class MetricsTracker:
    """
    Tracks token usage, costs, compression ratios, and cache hit rates.
    """
    
    def __init__(self, config: OMConfig):
        self.config = config
        self._threads: Dict[str, OMStats] = {}
        
    def _get_or_create_stats(self, thread_id: str) -> OMStats:
        if thread_id not in self._threads:
            self._threads[thread_id] = OMStats(thread_id=thread_id)
        return self._threads[thread_id]

    def record_context_build(self, thread_id: str, observation_tokens: int, message_tokens: int, total_tokens: int):
        stats = self._get_or_create_stats(thread_id)
        # Assuming observation tokens are what's stable and cached
        stats.total_cached_tokens += observation_tokens
        current_avg = stats.avg_context_window_tokens
        if current_avg == 0:
            stats.avg_context_window_tokens = total_tokens
        else:
            stats.avg_context_window_tokens = int((current_avg + total_tokens) / 2)
            
    def record_observer_run(self, thread_id: str, input_tokens: int, output_tokens: int, messages_compressed: int, observations_created: int):
        stats = self._get_or_create_stats(thread_id)
        stats.observer_runs += 1
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens
        stats.total_observations += observations_created
        
        # Recalculate cost
        self._recalculate_costs(stats)
        
    def record_reflector_run(self, thread_id: str, input_tokens: int, output_tokens: int, observations_before: int, observations_after: int):
        stats = self._get_or_create_stats(thread_id)
        stats.reflector_runs += 1
        stats.total_input_tokens += input_tokens
        stats.total_output_tokens += output_tokens
        
        # Track that observations changed
        self._recalculate_costs(stats)
        
    def record_cache_hit(self, thread_id: str, cached_tokens: int):
        stats = self._get_or_create_stats(thread_id)
        stats.total_cached_tokens += cached_tokens
        self._recalculate_costs(stats)
        
    def _recalculate_costs(self, stats: OMStats):
        if not self.config.track_costs:
            return
            
        in_cost = (stats.total_input_tokens / 1000) * self.config.cost_per_1k_input_tokens
        out_cost = (stats.total_output_tokens / 1000) * self.config.cost_per_1k_output_tokens
        
        # Approximate caching discount
        discount_value = (stats.total_cached_tokens / 1000) * self.config.cost_per_1k_input_tokens * self.config.cached_token_discount
        
        stats.estimated_cost_with_om = max(0.0, in_cost + out_cost - discount_value)
        stats.estimated_cost_without_om = self.estimate_rag_cost(stats.thread_id)
        stats.cost_savings = max(0.0, stats.estimated_cost_without_om - stats.estimated_cost_with_om)
        
        if stats.total_input_tokens > 0:
            stats.compression_ratio = stats.total_input_tokens / max(1, stats.total_output_tokens)
            
    def get_thread_stats(self, thread_id: str) -> OMStats:
        return self._get_or_create_stats(thread_id)
        
    def get_global_stats(self) -> dict:
        totals = {
            "total_threads": len(self._threads),
            "total_savings": sum(s.cost_savings for s in self._threads.values()),
            "total_tokens_saved": sum(s.total_cached_tokens for s in self._threads.values())
        }
        return totals
        
    def estimate_rag_cost(self, thread_id: str) -> float:
        """
        Estimate what the same conversation would cost with traditional RAG memory.
        Assumes: embedding per message, vector DB query per turn, reranking, no caching.
        """
        stats = self._get_or_create_stats(thread_id)
        
        # Rough estimation
        # Baseline context injected every turn continuously
        estimated_total_raw_tokens = stats.total_input_tokens * 2 # Simulated repeated injection
        
        # RAG doesn't have cached discount on changing retrieval chunks
        cost = (estimated_total_raw_tokens / 1000) * self.config.cost_per_1k_input_tokens
        # Vector search overhead + embedding (~10-20% extra)
        cost *= 1.15
        return cost
        
    def get_savings_report(self, thread_id: str) -> dict:
        stats = self._get_or_create_stats(thread_id)
        
        savings_percentage = 0.0
        if stats.estimated_cost_without_om > 0:
            savings_percentage = (stats.cost_savings / stats.estimated_cost_without_om) * 100
            
        return {
            "om_cost": round(stats.estimated_cost_with_om, 4),
            "estimated_rag_cost": round(stats.estimated_cost_without_om, 4),
            "savings_dollars": round(stats.cost_savings, 4),
            "savings_percentage": round(savings_percentage, 1),
            "compression_ratio": round(stats.compression_ratio, 1),
            "total_observations": stats.total_observations,
            "total_reflections": stats.reflector_runs,
            "messages_processed": stats.total_messages
        }
