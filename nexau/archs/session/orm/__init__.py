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

"""ORM layer for SQLModel."""

from .engine import DatabaseEngine, get_pk_fields, get_table_name
from .filters import (
    AndFilter,
    ComparisonFilter,
    Filter,
    FilterOperator,
    LogicalOperator,
    NotFilter,
    OrFilter,
    evaluate,
    from_query_string,
    to_query_string,
    to_sqlalchemy,
)
from .jsonl_engine import JSONLDatabaseEngine
from .memory_engine import InMemoryDatabaseEngine
from .remote_engine import RemoteDatabaseEngine
from .sql_engine import SQLDatabaseEngine

__all__ = [
    # DatabaseEngine classes
    "DatabaseEngine",
    "InMemoryDatabaseEngine",
    "SQLDatabaseEngine",
    "RemoteDatabaseEngine",
    "JSONLDatabaseEngine",
    # Filter DSL models
    "Filter",
    "ComparisonFilter",
    "AndFilter",
    "OrFilter",
    "NotFilter",
    "FilterOperator",
    "LogicalOperator",
    # Filter converter functions
    "to_sqlalchemy",
    "evaluate",
    "to_query_string",
    "from_query_string",
    # Utilities
    "get_pk_fields",
    "get_table_name",
]
