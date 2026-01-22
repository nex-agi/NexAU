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

"""Tests for the Filter Converter evaluate() function.

Feature: postgrest-filter-protocol
"""

from __future__ import annotations

import pytest

from nexau.archs.session.orm import (
    AndFilter,
    ComparisonFilter,
    FilterOperator,
    NotFilter,
    OrFilter,
    evaluate,
)

# ============================================================================
# Test Data
# ============================================================================

# Sample records for testing
ALICE = {"name": "alice", "age": 25, "email": "alice@example.com", "score": 85.5, "active": True, "role": "admin"}
BOB = {"name": "bob", "age": 30, "email": "bob@test.com", "score": 72.0, "active": True, "role": "user"}
CHARLIE = {"name": "charlie", "age": 35, "email": "charlie@example.com", "score": 90.5, "active": False, "role": "user"}
DIANA = {"name": "diana", "age": 28, "email": "diana@test.org", "score": 65.0, "active": True, "role": None}
ALICE_SMITH = {"name": "Alice Smith", "age": 22, "email": "asmith@example.com", "score": 88.0, "active": True, "role": "moderator"}


# ============================================================================
# ComparisonFilter Tests - Requirement 3.2-3.8
# ============================================================================


class TestComparisonFilterEvaluate:
    """Tests for ComparisonFilter evaluate function."""

    def test_eq_operator_string(self) -> None:
        """Test eq operator with string value (Requirement 3.2)."""
        filter_ = ComparisonFilter.eq("name", "alice")
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False

    def test_eq_operator_int(self) -> None:
        """Test eq operator with integer value (Requirement 3.2)."""
        filter_ = ComparisonFilter.eq("age", 25)
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False

    def test_eq_operator_float(self) -> None:
        """Test eq operator with float value (Requirement 3.2)."""
        filter_ = ComparisonFilter.eq("score", 85.5)
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False

    def test_eq_operator_bool(self) -> None:
        """Test eq operator with boolean value (Requirement 3.2)."""
        filter_ = ComparisonFilter.eq("active", False)
        assert evaluate(filter_, CHARLIE) is True
        assert evaluate(filter_, ALICE) is False

    def test_eq_operator_none(self) -> None:
        """Test eq operator with None value (Requirement 3.2)."""
        filter_ = ComparisonFilter.eq("role", None)
        assert evaluate(filter_, DIANA) is True
        assert evaluate(filter_, ALICE) is False

    def test_neq_operator(self) -> None:
        """Test neq operator (Requirement 3.3)."""
        filter_ = ComparisonFilter.neq("name", "alice")
        assert evaluate(filter_, ALICE) is False
        assert evaluate(filter_, BOB) is True
        assert evaluate(filter_, CHARLIE) is True

    def test_gt_operator(self) -> None:
        """Test gt operator (Requirement 3.4)."""
        filter_ = ComparisonFilter.gt("age", 28)
        assert evaluate(filter_, ALICE) is False  # 25 > 28 = False
        assert evaluate(filter_, BOB) is True  # 30 > 28 = True
        assert evaluate(filter_, CHARLIE) is True  # 35 > 28 = True
        assert evaluate(filter_, DIANA) is False  # 28 > 28 = False

    def test_gte_operator(self) -> None:
        """Test gte operator (Requirement 3.4)."""
        filter_ = ComparisonFilter.gte("age", 28)
        assert evaluate(filter_, ALICE) is False  # 25 >= 28 = False
        assert evaluate(filter_, BOB) is True  # 30 >= 28 = True
        assert evaluate(filter_, CHARLIE) is True  # 35 >= 28 = True
        assert evaluate(filter_, DIANA) is True  # 28 >= 28 = True

    def test_lt_operator(self) -> None:
        """Test lt operator (Requirement 3.4)."""
        filter_ = ComparisonFilter.lt("age", 28)
        assert evaluate(filter_, ALICE) is True  # 25 < 28 = True
        assert evaluate(filter_, BOB) is False  # 30 < 28 = False
        assert evaluate(filter_, DIANA) is False  # 28 < 28 = False

    def test_lte_operator(self) -> None:
        """Test lte operator (Requirement 3.4)."""
        filter_ = ComparisonFilter.lte("age", 28)
        assert evaluate(filter_, ALICE) is True  # 25 <= 28 = True
        assert evaluate(filter_, BOB) is False  # 30 <= 28 = False
        assert evaluate(filter_, DIANA) is True  # 28 <= 28 = True

    def test_like_operator_suffix_match(self) -> None:
        """Test like operator with suffix match (Requirement 3.5)."""
        filter_ = ComparisonFilter.like("email", "%@example.com")
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False
        assert evaluate(filter_, CHARLIE) is True

    def test_like_operator_prefix_match(self) -> None:
        """Test like operator with prefix match (Requirement 3.5)."""
        filter_ = ComparisonFilter.like("name", "a%")
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False
        assert evaluate(filter_, ALICE_SMITH) is False  # Case-sensitive

    def test_like_operator_contains_match(self) -> None:
        """Test like operator with contains match (Requirement 3.5)."""
        filter_ = ComparisonFilter.like("email", "%example%")
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False
        assert evaluate(filter_, CHARLIE) is True

    def test_like_operator_underscore_wildcard(self) -> None:
        """Test like operator with underscore wildcard (Requirement 3.5)."""
        filter_ = ComparisonFilter.like("name", "bo_")
        assert evaluate(filter_, BOB) is True
        assert evaluate(filter_, ALICE) is False

    def test_ilike_operator_case_insensitive(self) -> None:
        """Test ilike operator for case-insensitive matching (Requirement 3.6)."""
        filter_ = ComparisonFilter.ilike("name", "alice%")
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, ALICE_SMITH) is True  # Case-insensitive
        assert evaluate(filter_, BOB) is False

    def test_ilike_operator_mixed_case(self) -> None:
        """Test ilike operator with mixed case pattern (Requirement 3.6)."""
        filter_ = ComparisonFilter.ilike("name", "ALICE%")
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, ALICE_SMITH) is True

    def test_in_operator(self) -> None:
        """Test in operator (Requirement 3.7)."""
        filter_ = ComparisonFilter.in_("role", ["admin", "moderator"])
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False
        assert evaluate(filter_, ALICE_SMITH) is True

    def test_in_operator_empty_list(self) -> None:
        """Test in operator with empty list (Requirement 3.7)."""
        filter_ = ComparisonFilter.in_("role", [])
        assert evaluate(filter_, ALICE) is False
        assert evaluate(filter_, BOB) is False

    def test_in_operator_with_none(self) -> None:
        """Test in operator when field value is None (Requirement 3.7)."""
        filter_ = ComparisonFilter.in_("role", ["admin", "user"])
        assert evaluate(filter_, DIANA) is False  # role is None

    def test_is_null_operator(self) -> None:
        """Test is operator for NULL check (Requirement 3.8)."""
        filter_ = ComparisonFilter.is_null("role")
        assert evaluate(filter_, DIANA) is True
        assert evaluate(filter_, ALICE) is False


# ============================================================================
# LogicalFilter Tests - Requirement 3.9-3.11
# ============================================================================


class TestLogicalFilterEvaluate:
    """Tests for LogicalFilter evaluate function."""

    def test_and_filter(self) -> None:
        """Test AND filter (Requirement 3.9)."""
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.eq("active", True),
                ComparisonFilter.gt("age", 25),
            ]
        )
        assert evaluate(filter_, ALICE) is False  # active=True, age=25 (not > 25)
        assert evaluate(filter_, BOB) is True  # active=True, age=30
        assert evaluate(filter_, CHARLIE) is False  # active=False
        assert evaluate(filter_, DIANA) is True  # active=True, age=28

    def test_and_filter_empty(self) -> None:
        """Test AND filter with empty filters list (Requirement 3.9)."""
        filter_ = AndFilter(filters=[])
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is True

    def test_or_filter(self) -> None:
        """Test OR filter (Requirement 3.10)."""
        filter_ = OrFilter(
            filters=[
                ComparisonFilter.eq("role", "admin"),
                ComparisonFilter.eq("role", "moderator"),
            ]
        )
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False
        assert evaluate(filter_, ALICE_SMITH) is True

    def test_or_filter_empty(self) -> None:
        """Test OR filter with empty filters list (Requirement 3.10)."""
        filter_ = OrFilter(filters=[])
        assert evaluate(filter_, ALICE) is False
        assert evaluate(filter_, BOB) is False

    def test_not_filter(self) -> None:
        """Test NOT filter (Requirement 3.11)."""
        filter_ = NotFilter(filter=ComparisonFilter.eq("active", True))
        assert evaluate(filter_, ALICE) is False
        assert evaluate(filter_, CHARLIE) is True

    def test_nested_logical_filters(self) -> None:
        """Test nested logical filters."""
        # (active = True AND age > 25) OR role = 'admin'
        filter_ = OrFilter(
            filters=[
                AndFilter(
                    filters=[
                        ComparisonFilter.eq("active", True),
                        ComparisonFilter.gt("age", 25),
                    ]
                ),
                ComparisonFilter.eq("role", "admin"),
            ]
        )
        assert evaluate(filter_, ALICE) is True  # role = admin
        assert evaluate(filter_, BOB) is True  # active=True, age=30
        assert evaluate(filter_, CHARLIE) is False  # active=False, role=user
        assert evaluate(filter_, DIANA) is True  # active=True, age=28

    def test_deeply_nested_filters(self) -> None:
        """Test deeply nested filters."""
        # NOT (role = 'user' OR (active = False AND age > 30))
        filter_ = NotFilter(
            filter=OrFilter(
                filters=[
                    ComparisonFilter.eq("role", "user"),
                    AndFilter(
                        filters=[
                            ComparisonFilter.eq("active", False),
                            ComparisonFilter.gt("age", 30),
                        ]
                    ),
                ]
            )
        )
        assert evaluate(filter_, ALICE) is True  # role=admin, not user
        assert evaluate(filter_, BOB) is False  # role=user
        assert evaluate(filter_, CHARLIE) is False  # role=user (also inactive and age > 30)
        assert evaluate(filter_, DIANA) is True  # role=None, not user
        assert evaluate(filter_, ALICE_SMITH) is True  # role=moderator


# ============================================================================
# Missing Field Tests - Requirement 3.12
# ============================================================================


class TestMissingFieldEvaluate:
    """Tests for missing field handling (Requirement 3.12)."""

    def test_missing_field_treated_as_none(self) -> None:
        """Test that missing field is treated as None (Requirement 3.12)."""
        record = {"name": "test"}  # No 'age' field
        filter_ = ComparisonFilter.eq("age", None)
        assert evaluate(filter_, record) is True

    def test_missing_field_eq_value_returns_false(self) -> None:
        """Test that missing field compared to non-None value returns False."""
        record = {"name": "test"}  # No 'age' field
        filter_ = ComparisonFilter.eq("age", 25)
        assert evaluate(filter_, record) is False

    def test_missing_field_comparison_returns_false(self) -> None:
        """Test that comparison operators with missing field return False."""
        record = {"name": "test"}  # No 'age' field

        assert evaluate(ComparisonFilter.gt("age", 25), record) is False
        assert evaluate(ComparisonFilter.gte("age", 25), record) is False
        assert evaluate(ComparisonFilter.lt("age", 25), record) is False
        assert evaluate(ComparisonFilter.lte("age", 25), record) is False

    def test_missing_field_like_returns_false(self) -> None:
        """Test that like operator with missing field returns False."""
        record = {"name": "test"}  # No 'email' field
        filter_ = ComparisonFilter.like("email", "%@example.com")
        assert evaluate(filter_, record) is False

    def test_missing_field_in_returns_false(self) -> None:
        """Test that in operator with missing field returns False."""
        record = {"name": "test"}  # No 'role' field
        filter_ = ComparisonFilter.in_("role", ["admin", "user"])
        assert evaluate(filter_, record) is False

    def test_missing_field_is_null_returns_true(self) -> None:
        """Test that is_null with missing field returns True."""
        record = {"name": "test"}  # No 'role' field
        filter_ = ComparisonFilter.is_null("role")
        assert evaluate(filter_, record) is True


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEvaluateEdgeCases:
    """Tests for edge cases in evaluate function."""

    def test_single_filter_in_and(self) -> None:
        """Test AndFilter with single filter."""
        filter_ = AndFilter(filters=[ComparisonFilter.eq("name", "alice")])
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False

    def test_single_filter_in_or(self) -> None:
        """Test OrFilter with single filter."""
        filter_ = OrFilter(filters=[ComparisonFilter.eq("name", "bob")])
        assert evaluate(filter_, BOB) is True
        assert evaluate(filter_, ALICE) is False

    def test_double_negation(self) -> None:
        """Test double negation (NOT NOT)."""
        filter_ = NotFilter(filter=NotFilter(filter=ComparisonFilter.eq("name", "alice")))
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False

    def test_like_with_special_regex_chars(self) -> None:
        """Test LIKE with special regex characters in pattern."""
        record = {"email": "test.user@example.com"}
        # The dot should be treated literally, not as regex wildcard
        filter_ = ComparisonFilter.like("email", "test.%")
        assert evaluate(filter_, record) is True

        # Without the dot
        filter2 = ComparisonFilter.like("email", "testX%")
        assert evaluate(filter2, record) is False

    def test_like_exact_match(self) -> None:
        """Test LIKE with exact match (no wildcards)."""
        filter_ = ComparisonFilter.like("name", "alice")
        assert evaluate(filter_, ALICE) is True
        assert evaluate(filter_, BOB) is False

    def test_comparison_with_none_value(self) -> None:
        """Test comparison operators when filter value is None."""
        filter_gt = ComparisonFilter.gt("age", None)  # type: ignore
        assert evaluate(filter_gt, ALICE) is False

    def test_neq_with_none(self) -> None:
        """Test neq operator with None values."""
        filter_ = ComparisonFilter.neq("role", None)
        assert evaluate(filter_, ALICE) is True  # role="admin" != None
        assert evaluate(filter_, DIANA) is False  # role=None != None is False

    def test_in_operator_with_invalid_value_type(self) -> None:
        """Test in operator with non-list value raises ValueError."""
        filter_ = ComparisonFilter(field="role", op=FilterOperator.IN, value="not_a_list")
        with pytest.raises(ValueError) as exc_info:
            evaluate(filter_, ALICE)
        assert "Invalid value type for operator" in str(exc_info.value)
