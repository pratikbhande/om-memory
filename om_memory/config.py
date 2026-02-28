import os
from typing import Optional

from pydantic import ValidationError
from om_memory.models import OMConfig

def from_env() -> OMConfig:
    """
    Load OMConfig from environment variables using a predefined mapping.
    Users can override any setting by prefixing the env var with `OM_`.
    """
    config_kwargs = {}
    
    # Thresholds
    if "OM_OBSERVER_THRESHOLD" in os.environ:
        config_kwargs["observer_token_threshold"] = int(os.environ["OM_OBSERVER_THRESHOLD"])
    if "OM_REFLECTOR_THRESHOLD" in os.environ:
        config_kwargs["reflector_token_threshold"] = int(os.environ["OM_REFLECTOR_THRESHOLD"])
    if "OM_MAX_MESSAGE_HISTORY" in os.environ:
        config_kwargs["max_message_history_tokens"] = int(os.environ["OM_MAX_MESSAGE_HISTORY"])
        
    # Models
    if "OM_OBSERVER_MODEL" in os.environ:
        config_kwargs["observer_model"] = os.environ["OM_OBSERVER_MODEL"]
    if "OM_REFLECTOR_MODEL" in os.environ:
        config_kwargs["reflector_model"] = os.environ["OM_REFLECTOR_MODEL"]
        
    # Behavior
    if "OM_AUTO_OBSERVE" in os.environ:
        config_kwargs["auto_observe"] = os.environ["OM_AUTO_OBSERVE"].lower() in ("true", "1", "yes")
    if "OM_AUTO_REFLECT" in os.environ:
        config_kwargs["auto_reflect"] = os.environ["OM_AUTO_REFLECT"].lower() in ("true", "1", "yes")
    if "OM_BLOCKING_MODE" in os.environ:
        config_kwargs["blocking_mode"] = os.environ["OM_BLOCKING_MODE"].lower() in ("true", "1", "yes")

    # Tracking
    if "OM_TRACK_COSTS" in os.environ:
        config_kwargs["track_costs"] = os.environ["OM_TRACK_COSTS"].lower() in ("true", "1", "yes")
    
    # New settings
    if "OM_DEMO_MODE" in os.environ:
        config_kwargs["demo_mode"] = os.environ["OM_DEMO_MODE"].lower() in ("true", "1", "yes")
    if "OM_MESSAGE_RETENTION" in os.environ:
        config_kwargs["message_retention_count"] = int(os.environ["OM_MESSAGE_RETENTION"])
    if "OM_MESSAGE_TOKEN_BUDGET" in os.environ:
        config_kwargs["message_token_budget"] = int(os.environ["OM_MESSAGE_TOKEN_BUDGET"])
    if "OM_SHARE_TOKEN_BUDGET" in os.environ:
        config_kwargs["share_token_budget"] = os.environ["OM_SHARE_TOKEN_BUDGET"].lower() in ("true", "1", "yes")
        
    return OMConfig(**config_kwargs)

# Global default config loaded from env
default_config = from_env()
