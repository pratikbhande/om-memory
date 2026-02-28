"""
Shared observation parsing utility.

Extracts structured Observation objects from LLM responses.
Used by both Observer and Reflector to avoid code duplication.
"""

import re
import uuid
from datetime import datetime, timezone

from om_memory.models import Observation, Priority


def parse_observations(
    llm_response: str,
    thread_id: str,
    source_message_ids: list[str],
    resource_id: str = None,
) -> list[Observation]:
    """
    Parse an LLM response into a list of Observation objects.

    Handles:
    - CURRENT_TASK / SUGGESTED_NEXT extraction
    - Date: YYYY-MM-DD grouping headers
    - Priority emoji lines (🔴 🟡 🟢)
    - Referenced date and relative date annotations

    Args:
        llm_response: Raw text from the LLM.
        thread_id: The thread these observations belong to.
        source_message_ids: IDs of messages that were compressed.
        resource_id: Optional resource ID for resource-scoped memory.

    Returns:
        List of parsed Observation objects.
    """
    observations: list[Observation] = []
    current_date = datetime.now(timezone.utc)

    # Extract CURRENT_TASK and SUGGESTED_NEXT
    task_match = re.search(r"CURRENT_TASK:\s*(.*)", llm_response)
    next_match = re.search(r"SUGGESTED_NEXT:\s*(.*)", llm_response)

    if task_match:
        observations.append(Observation(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            resource_id=resource_id,
            priority=Priority.CRITICAL,
            content=f"CURRENT TASK: {task_match.group(1).strip()}",
            source_message_ids=source_message_ids,
        ))

    if next_match:
        observations.append(Observation(
            id=str(uuid.uuid4()),
            thread_id=thread_id,
            resource_id=resource_id,
            priority=Priority.IMPORTANT,
            content=f"SUGGESTED NEXT: {next_match.group(1).strip()}",
            source_message_ids=source_message_ids,
        ))

    lines = llm_response.split("\n")

    for line in lines:
        line = line.strip()
        if line.startswith("Date:"):
            try:
                date_str = line.split(":", 1)[1].strip()
                parsed_date = datetime.strptime(date_str, "%Y-%m-%d").date()
                current_date = datetime.combine(
                    parsed_date, current_date.time(), tzinfo=timezone.utc
                )
            except Exception:
                pass  # Ignore parse errors, keep current utc
            continue

        if line.startswith("-") and any(
            emoji in line for emoji in ["🔴", "🟡", "🟢"]
        ):
            try:
                # Determine priority
                priority_val = Priority.INFO
                for p in Priority:
                    if p.value in line:
                        priority_val = p
                        break

                # Extract time and content
                time_match = re.search(r"(\d{2}:\d{2})", line)
                obs_time = current_date.time()
                if time_match:
                    obs_time = datetime.strptime(
                        time_match.group(1), "%H:%M"
                    ).time()

                content_start = line.find(priority_val.value) + len(
                    priority_val.value
                )
                if time_match:
                    content_start = line.find(time_match.group(1)) + len(
                        time_match.group(1)
                    )

                raw_content = line[content_start:].strip()

                # Extract references
                ref_date = None
                rel_date = None
                ref_match = re.search(r"\(([^)]*referenced[^)]*)\)", raw_content)
                if ref_match:
                    ref_str = ref_match.group(1)
                    raw_content = raw_content.replace(f"({ref_str})", "").strip()

                    date_match = re.search(
                        r"referenced:\s*(\d{4}-\d{2}-\d{2})", ref_str
                    )
                    if date_match:
                        try:
                            ref_date = datetime.strptime(
                                date_match.group(1), "%Y-%m-%d"
                            ).replace(tzinfo=timezone.utc)
                        except Exception:
                            pass

                    meaning_match = re.search(
                        r'meaning\s*"([^"]+)"', ref_str
                    )
                    if meaning_match:
                        rel_date = meaning_match.group(1)

                obs_date = datetime.combine(
                    current_date.date(), obs_time, tzinfo=timezone.utc
                )

                obs = Observation(
                    id=str(uuid.uuid4()),
                    thread_id=thread_id,
                    resource_id=resource_id,
                    observation_date=obs_date,
                    referenced_date=ref_date,
                    relative_date=rel_date,
                    priority=priority_val,
                    content=raw_content,
                    source_message_ids=source_message_ids,
                )
                observations.append(obs)
            except Exception:
                # Ignore malformed lines gracefully
                pass

    return observations
