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

from __future__ import annotations

import json
import logging
import uuid
from typing import Any, cast

logger = logging.getLogger(__name__)

_MAX_DEPTH = 50  # Maximum recursion depth for serialization


def sanitize_for_serialization(obj: Any, _depth: int = 0) -> Any:
    # Guard against infinite recursion
    if _depth > _MAX_DEPTH:
        raise ValueError(f"Maximum serialization depth ({_MAX_DEPTH}) exceeded")

    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    try:
        if hasattr(obj, "model_dump"):
            return sanitize_for_serialization(obj.model_dump(mode="json"), _depth + 1)
        if hasattr(obj, "dict"):
            return sanitize_for_serialization(obj.dict(), _depth + 1)
    except RecursionError:
        raise ValueError(f"Circular reference detected in object of type {type(obj)}")

    if isinstance(obj, dict):
        clean: dict[str, Any] = {}
        for k, v in cast(dict[Any, Any], obj).items():
            try:
                clean[cast(str, k)] = sanitize_for_serialization(v, _depth + 1)
            except (ValueError, TypeError, RecursionError) as e:
                logger.debug("Skipping non-serializable dict value for key '%s': %s", k, e)
        return clean

    if isinstance(obj, (list, tuple)):
        clean_list: list[Any] = []
        for i, v in enumerate(cast(list[Any], obj)):
            try:
                clean_list.append(sanitize_for_serialization(v, _depth + 1))
            except (ValueError, TypeError, RecursionError) as e:
                logger.debug("Skipping non-serializable list item at index %d: %s", i, e)
        return clean_list

    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    if isinstance(obj, uuid.UUID):
        return str(obj)

    try:
        json.dumps(obj)
        return obj
    except Exception:
        raise ValueError(f"Object of type {type(obj)} is not serializable")
