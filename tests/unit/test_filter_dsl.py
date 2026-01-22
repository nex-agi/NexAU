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

"""Tests for the Filter DSL models."""

from __future__ import annotations

from nexau.archs.session.orm import (
    AndFilter,
    ComparisonFilter,
    FilterOperator,
    LogicalOperator,
    NotFilter,
    OrFilter,
)
from nexau.archs.session.orm.filters.dsl import FilterBase

# ============================================================================
# FilterOperator Tests
# ============================================================================


class TestFilterOperator:
    """Tests for FilterOperator enum."""

    def test_all_operators_defined(self) -> None:
        """Test that all required operators are defined."""
        expected_operators = ["eq", "neq", "gt", "gte", "lt", "lte", "like", "ilike", "in", "is"]
        actual_operators = [op.value for op in FilterOperator]
        assert set(expected_operators) == set(actual_operators)

    def test_operator_string_values(self) -> None:
        """Test that operators have correct string values."""
        assert FilterOperator.EQ.value == "eq"
        assert FilterOperator.NEQ.value == "neq"
        assert FilterOperator.GT.value == "gt"
        assert FilterOperator.GTE.value == "gte"
        assert FilterOperator.LT.value == "lt"
        assert FilterOperator.LTE.value == "lte"
        assert FilterOperator.LIKE.value == "like"
        assert FilterOperator.ILIKE.value == "ilike"
        assert FilterOperator.IN.value == "in"
        assert FilterOperator.IS.value == "is"


class TestLogicalOperator:
    """Tests for LogicalOperator enum."""

    def test_all_logical_operators_defined(self) -> None:
        """Test that all required logical operators are defined."""
        expected_operators = ["and", "or", "not"]
        actual_operators = [op.value for op in LogicalOperator]
        assert set(expected_operators) == set(actual_operators)


# ============================================================================
# ComparisonFilter Tests
# ============================================================================


class TestComparisonFilter:
    """Tests for ComparisonFilter model."""

    def test_comparison_filter_attributes(self) -> None:
        """Test that ComparisonFilter has all required attributes."""
        filter_ = ComparisonFilter(field="name", op=FilterOperator.EQ, value="alice")
        assert filter_.type == "comparison"
        assert filter_.field == "name"
        assert filter_.op == FilterOperator.EQ
        assert filter_.value == "alice"

    def test_comparison_filter_inherits_from_filter_base(self) -> None:
        """Test that ComparisonFilter inherits from FilterBase."""
        assert issubclass(ComparisonFilter, FilterBase)

    def test_comparison_filter_json_serialization(self) -> None:
        """Test JSON serialization of ComparisonFilter."""
        filter_ = ComparisonFilter(field="age", op=FilterOperator.GT, value=18)
        json_str = filter_.model_dump_json()
        assert '"type":"comparison"' in json_str
        assert '"field":"age"' in json_str
        assert '"op":"gt"' in json_str
        assert '"value":18' in json_str

    def test_comparison_filter_json_deserialization(self) -> None:
        """Test JSON deserialization of ComparisonFilter."""
        json_str = '{"type":"comparison","field":"status","op":"eq","value":"active"}'
        filter_ = ComparisonFilter.model_validate_json(json_str)
        assert filter_.type == "comparison"
        assert filter_.field == "status"
        assert filter_.op == FilterOperator.EQ
        assert filter_.value == "active"

    def test_comparison_filter_value_types(self) -> None:
        """Test that ComparisonFilter accepts various value types."""
        # String value
        f1 = ComparisonFilter(field="name", op=FilterOperator.EQ, value="alice")
        assert f1.value == "alice"

        # Integer value
        f2 = ComparisonFilter(field="age", op=FilterOperator.GT, value=18)
        assert f2.value == 18

        # Float value
        f3 = ComparisonFilter(field="price", op=FilterOperator.LTE, value=99.99)
        assert f3.value == 99.99

        # Boolean value
        f4 = ComparisonFilter(field="active", op=FilterOperator.EQ, value=True)
        assert f4.value is True

        # None value
        f5 = ComparisonFilter(field="deleted_at", op=FilterOperator.IS, value=None)
        assert f5.value is None

        # List value
        f6 = ComparisonFilter(field="status", op=FilterOperator.IN, value=["active", "pending"])
        assert f6.value == ["active", "pending"]


# ============================================================================
# ComparisonFilter Factory Methods Tests
# ============================================================================


class TestComparisonFilterFactoryMethods:
    """Tests for ComparisonFilter convenience factory methods."""

    def test_eq_factory_method(self) -> None:
        """Test eq() factory method."""
        filter_ = ComparisonFilter.eq("name", "alice")
        assert filter_.type == "comparison"
        assert filter_.field == "name"
        assert filter_.op == FilterOperator.EQ
        assert filter_.value == "alice"

    def test_neq_factory_method(self) -> None:
        """Test neq() factory method."""
        filter_ = ComparisonFilter.neq("status", "deleted")
        assert filter_.type == "comparison"
        assert filter_.field == "status"
        assert filter_.op == FilterOperator.NEQ
        assert filter_.value == "deleted"

    def test_gt_factory_method(self) -> None:
        """Test gt() factory method."""
        filter_ = ComparisonFilter.gt("age", 18)
        assert filter_.type == "comparison"
        assert filter_.field == "age"
        assert filter_.op == FilterOperator.GT
        assert filter_.value == 18

    def test_gte_factory_method(self) -> None:
        """Test gte() factory method."""
        filter_ = ComparisonFilter.gte("score", 60)
        assert filter_.type == "comparison"
        assert filter_.field == "score"
        assert filter_.op == FilterOperator.GTE
        assert filter_.value == 60

    def test_lt_factory_method(self) -> None:
        """Test lt() factory method."""
        filter_ = ComparisonFilter.lt("price", 100)
        assert filter_.type == "comparison"
        assert filter_.field == "price"
        assert filter_.op == FilterOperator.LT
        assert filter_.value == 100

    def test_lte_factory_method(self) -> None:
        """Test lte() factory method."""
        filter_ = ComparisonFilter.lte("quantity", 10)
        assert filter_.type == "comparison"
        assert filter_.field == "quantity"
        assert filter_.op == FilterOperator.LTE
        assert filter_.value == 10

    def test_like_factory_method(self) -> None:
        """Test like() factory method."""
        filter_ = ComparisonFilter.like("name", "%alice%")
        assert filter_.type == "comparison"
        assert filter_.field == "name"
        assert filter_.op == FilterOperator.LIKE
        assert filter_.value == "%alice%"

    def test_ilike_factory_method(self) -> None:
        """Test ilike() factory method."""
        filter_ = ComparisonFilter.ilike("email", "%@example.com")
        assert filter_.type == "comparison"
        assert filter_.field == "email"
        assert filter_.op == FilterOperator.ILIKE
        assert filter_.value == "%@example.com"

    def test_in_factory_method(self) -> None:
        """Test in_() factory method."""
        filter_ = ComparisonFilter.in_("status", ["active", "pending", "review"])
        assert filter_.type == "comparison"
        assert filter_.field == "status"
        assert filter_.op == FilterOperator.IN
        assert filter_.value == ["active", "pending", "review"]

    def test_is_null_factory_method(self) -> None:
        """Test is_null() factory method."""
        filter_ = ComparisonFilter.is_null("deleted_at")
        assert filter_.type == "comparison"
        assert filter_.field == "deleted_at"
        assert filter_.op == FilterOperator.IS
        assert filter_.value is None

    def test_factory_methods_with_different_value_types(self) -> None:
        """Test factory methods with different value types."""
        # eq with different types
        assert ComparisonFilter.eq("count", 42).value == 42
        assert ComparisonFilter.eq("ratio", 3.14).value == 3.14
        assert ComparisonFilter.eq("active", True).value is True
        assert ComparisonFilter.eq("name", None).value is None

        # in_ with different list types
        assert ComparisonFilter.in_("ids", [1, 2, 3]).value == [1, 2, 3]
        assert ComparisonFilter.in_("prices", [1.5, 2.5]).value == [1.5, 2.5]


# ============================================================================
# JSON Round-trip Tests
# ============================================================================


class TestComparisonFilterJsonRoundTrip:
    """Tests for JSON serialization round-trip consistency."""

    def test_json_roundtrip_string_value(self) -> None:
        """Test JSON round-trip with string value."""
        original = ComparisonFilter.eq("name", "alice")
        json_str = original.model_dump_json()
        restored = ComparisonFilter.model_validate_json(json_str)
        assert original == restored

    def test_json_roundtrip_int_value(self) -> None:
        """Test JSON round-trip with integer value."""
        original = ComparisonFilter.gt("age", 18)
        json_str = original.model_dump_json()
        restored = ComparisonFilter.model_validate_json(json_str)
        assert original == restored

    def test_json_roundtrip_float_value(self) -> None:
        """Test JSON round-trip with float value."""
        original = ComparisonFilter.lte("price", 99.99)
        json_str = original.model_dump_json()
        restored = ComparisonFilter.model_validate_json(json_str)
        assert original == restored

    def test_json_roundtrip_bool_value(self) -> None:
        """Test JSON round-trip with boolean value."""
        original = ComparisonFilter.eq("active", True)
        json_str = original.model_dump_json()
        restored = ComparisonFilter.model_validate_json(json_str)
        assert original == restored

    def test_json_roundtrip_null_value(self) -> None:
        """Test JSON round-trip with null value."""
        original = ComparisonFilter.is_null("deleted_at")
        json_str = original.model_dump_json()
        restored = ComparisonFilter.model_validate_json(json_str)
        assert original == restored

    def test_json_roundtrip_list_value(self) -> None:
        """Test JSON round-trip with list value."""
        original = ComparisonFilter.in_("status", ["active", "pending"])
        json_str = original.model_dump_json()
        restored = ComparisonFilter.model_validate_json(json_str)
        assert original == restored

    def test_json_roundtrip_all_operators(self) -> None:
        """Test JSON round-trip for all operators."""
        filters = [
            ComparisonFilter.eq("f", "v"),
            ComparisonFilter.neq("f", "v"),
            ComparisonFilter.gt("f", 1),
            ComparisonFilter.gte("f", 1),
            ComparisonFilter.lt("f", 1),
            ComparisonFilter.lte("f", 1),
            ComparisonFilter.like("f", "%v%"),
            ComparisonFilter.ilike("f", "%v%"),
            ComparisonFilter.in_("f", ["a", "b"]),
            ComparisonFilter.is_null("f"),
        ]
        for original in filters:
            json_str = original.model_dump_json()
            restored = ComparisonFilter.model_validate_json(json_str)
            assert original == restored, f"Round-trip failed for operator {original.op}"


# ============================================================================
# AndFilter Tests
# ============================================================================


class TestAndFilter:
    """Tests for AndFilter model."""

    def test_and_filter_attributes(self) -> None:
        """Test that AndFilter has all required attributes."""
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.eq("name", "alice"),
                ComparisonFilter.gt("age", 18),
            ]
        )
        assert filter_.type == "and"
        assert len(filter_.filters) == 2

    def test_and_filter_inherits_from_filter_base(self) -> None:
        """Test that AndFilter inherits from FilterBase."""
        assert issubclass(AndFilter, FilterBase)

    def test_and_filter_empty_filters(self) -> None:
        """Test AndFilter with empty filters list."""
        filter_ = AndFilter(filters=[])
        assert filter_.type == "and"
        assert filter_.filters == []

    def test_and_filter_json_serialization(self) -> None:
        """Test JSON serialization of AndFilter."""
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.eq("status", "active"),
                ComparisonFilter.gte("score", 60),
            ]
        )
        json_str = filter_.model_dump_json()
        assert '"type":"and"' in json_str
        assert '"filters"' in json_str

    def test_and_filter_json_deserialization(self) -> None:
        """Test JSON deserialization of AndFilter."""
        json_str = '{"type":"and","filters":[{"type":"comparison","field":"name","op":"eq","value":"alice"}]}'
        filter_ = AndFilter.model_validate_json(json_str)
        assert filter_.type == "and"
        assert len(filter_.filters) == 1
        first_filter = filter_.filters[0]
        assert isinstance(first_filter, ComparisonFilter)
        assert first_filter.field == "name"

    def test_and_filter_nested_logical_filters(self) -> None:
        """Test AndFilter containing nested logical filters."""
        inner_or = OrFilter(
            filters=[
                ComparisonFilter.eq("role", "admin"),
                ComparisonFilter.eq("role", "moderator"),
            ]
        )
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.eq("active", True),
                inner_or,
            ]
        )
        assert filter_.type == "and"
        assert len(filter_.filters) == 2
        assert isinstance(filter_.filters[1], OrFilter)


# ============================================================================
# OrFilter Tests
# ============================================================================


class TestOrFilter:
    """Tests for OrFilter model."""

    def test_or_filter_attributes(self) -> None:
        """Test that OrFilter has all required attributes."""
        filter_ = OrFilter(
            filters=[
                ComparisonFilter.eq("status", "active"),
                ComparisonFilter.eq("status", "pending"),
            ]
        )
        assert filter_.type == "or"
        assert len(filter_.filters) == 2

    def test_or_filter_inherits_from_filter_base(self) -> None:
        """Test that OrFilter inherits from FilterBase."""
        assert issubclass(OrFilter, FilterBase)

    def test_or_filter_empty_filters(self) -> None:
        """Test OrFilter with empty filters list."""
        filter_ = OrFilter(filters=[])
        assert filter_.type == "or"
        assert filter_.filters == []

    def test_or_filter_json_serialization(self) -> None:
        """Test JSON serialization of OrFilter."""
        filter_ = OrFilter(
            filters=[
                ComparisonFilter.eq("role", "admin"),
                ComparisonFilter.eq("role", "user"),
            ]
        )
        json_str = filter_.model_dump_json()
        assert '"type":"or"' in json_str
        assert '"filters"' in json_str

    def test_or_filter_json_deserialization(self) -> None:
        """Test JSON deserialization of OrFilter."""
        json_str = '{"type":"or","filters":[{"type":"comparison","field":"status","op":"eq","value":"active"}]}'
        filter_ = OrFilter.model_validate_json(json_str)
        assert filter_.type == "or"
        assert len(filter_.filters) == 1
        first_filter = filter_.filters[0]
        assert isinstance(first_filter, ComparisonFilter)
        assert first_filter.field == "status"

    def test_or_filter_nested_logical_filters(self) -> None:
        """Test OrFilter containing nested logical filters."""
        inner_and = AndFilter(
            filters=[
                ComparisonFilter.eq("active", True),
                ComparisonFilter.gte("age", 18),
            ]
        )
        filter_ = OrFilter(
            filters=[
                ComparisonFilter.eq("role", "admin"),
                inner_and,
            ]
        )
        assert filter_.type == "or"
        assert len(filter_.filters) == 2
        assert isinstance(filter_.filters[1], AndFilter)


# ============================================================================
# NotFilter Tests
# ============================================================================


class TestNotFilter:
    """Tests for NotFilter model."""

    def test_not_filter_attributes(self) -> None:
        """Test that NotFilter has all required attributes."""
        filter_ = NotFilter(filter=ComparisonFilter.eq("status", "deleted"))
        assert filter_.type == "not"
        inner_filter = filter_.filter
        assert isinstance(inner_filter, ComparisonFilter)
        assert inner_filter.field == "status"

    def test_not_filter_inherits_from_filter_base(self) -> None:
        """Test that NotFilter inherits from FilterBase."""
        assert issubclass(NotFilter, FilterBase)

    def test_not_filter_json_serialization(self) -> None:
        """Test JSON serialization of NotFilter."""
        filter_ = NotFilter(filter=ComparisonFilter.eq("status", "deleted"))
        json_str = filter_.model_dump_json()
        assert '"type":"not"' in json_str
        assert '"filter"' in json_str

    def test_not_filter_json_deserialization(self) -> None:
        """Test JSON deserialization of NotFilter."""
        json_str = '{"type":"not","filter":{"type":"comparison","field":"status","op":"eq","value":"deleted"}}'
        filter_ = NotFilter.model_validate_json(json_str)
        assert filter_.type == "not"
        inner_filter = filter_.filter
        assert isinstance(inner_filter, ComparisonFilter)
        assert inner_filter.field == "status"
        assert inner_filter.value == "deleted"

    def test_not_filter_with_logical_filter(self) -> None:
        """Test NotFilter containing a logical filter."""
        inner_or = OrFilter(
            filters=[
                ComparisonFilter.eq("status", "deleted"),
                ComparisonFilter.eq("status", "archived"),
            ]
        )
        filter_ = NotFilter(filter=inner_or)
        assert filter_.type == "not"
        assert isinstance(filter_.filter, OrFilter)

    def test_not_filter_nested_not(self) -> None:
        """Test NotFilter containing another NotFilter (double negation)."""
        inner_not = NotFilter(filter=ComparisonFilter.eq("active", False))
        filter_ = NotFilter(filter=inner_not)
        assert filter_.type == "not"
        assert isinstance(filter_.filter, NotFilter)


# ============================================================================
# Logical Filter JSON Round-trip Tests
# ============================================================================


class TestLogicalFilterJsonRoundTrip:
    """Tests for JSON serialization round-trip consistency of logical filters."""

    def test_and_filter_json_roundtrip(self) -> None:
        """Test JSON round-trip for AndFilter."""
        original = AndFilter(
            filters=[
                ComparisonFilter.eq("name", "alice"),
                ComparisonFilter.gt("age", 18),
            ]
        )
        json_str = original.model_dump_json()
        restored = AndFilter.model_validate_json(json_str)
        assert original == restored

    def test_or_filter_json_roundtrip(self) -> None:
        """Test JSON round-trip for OrFilter."""
        original = OrFilter(
            filters=[
                ComparisonFilter.eq("status", "active"),
                ComparisonFilter.eq("status", "pending"),
            ]
        )
        json_str = original.model_dump_json()
        restored = OrFilter.model_validate_json(json_str)
        assert original == restored

    def test_not_filter_json_roundtrip(self) -> None:
        """Test JSON round-trip for NotFilter."""
        original = NotFilter(filter=ComparisonFilter.eq("status", "deleted"))
        json_str = original.model_dump_json()
        restored = NotFilter.model_validate_json(json_str)
        assert original == restored

    def test_deeply_nested_filter_json_roundtrip(self) -> None:
        """Test JSON round-trip for deeply nested filters."""
        original = AndFilter(
            filters=[
                OrFilter(
                    filters=[
                        ComparisonFilter.eq("role", "admin"),
                        AndFilter(
                            filters=[
                                ComparisonFilter.eq("role", "user"),
                                ComparisonFilter.gte("level", 5),
                            ]
                        ),
                    ]
                ),
                NotFilter(filter=ComparisonFilter.eq("status", "banned")),
            ]
        )
        json_str = original.model_dump_json()
        restored = AndFilter.model_validate_json(json_str)
        assert original == restored
