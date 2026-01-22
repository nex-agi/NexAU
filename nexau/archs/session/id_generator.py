# Copyright (c) Nex-AGI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""ID generation utilities using ULID with type prefixes.

ULID (Universally Unique Lexicographically Sortable Identifier):
- 128-bit identifier (vs UUID's 128 bits)
- Time-sortable (first 48 bits are timestamp)
- Case-insensitive Base32 encoding (26 characters)
- Collision-resistant (80 bits of randomness per millisecond)
- URL-safe and human-readable

Format: {prefix}_{ULID}
Example: sess_01HQZX3Y4K5M6N7P8Q9R0S1T2V

Note: agent_id uses shorter UUID format for easier copying by agents.
"""

from __future__ import annotations

import uuid

from ulid import ULID


def generate_session_id() -> str:
    """Generate a session ID with 'sess_' prefix.

    Returns:
        Session ID in format: sess_{ULID}
        Example: sess_01HQZX3Y4K5M6N7P8Q9R0S1T2V
    """
    return f"sess_{ULID()}"


def generate_agent_id() -> str:
    """Generate an agent ID with 'agent_' prefix.

    Uses shorter UUID format (8 hex chars = 32 bits) for easier copying.
    Since agent_id is scoped within a session, collision risk is acceptable.

    Returns:
        Agent ID in format: agent_{8_hex_chars}
        Example: agent_a1b2c3d4
    """
    return f"agent_{uuid.uuid4().hex[:8]}"


def generate_run_id() -> str:
    """Generate a run ID with 'run_' prefix.

    Returns:
        Run ID in format: run_{ULID}
        Example: run_01HQZX3Y4K5M6N7P8Q9R0S1T2V
    """
    return f"run_{ULID()}"


def generate_action_id() -> str:
    """Generate an action ID with 'act_' prefix.

    Returns:
        Action ID in format: act_{ULID}
        Example: act_01HQZX3Y4K5M6N7P8Q9R0S1T2V
    """
    return f"act_{ULID()}"
