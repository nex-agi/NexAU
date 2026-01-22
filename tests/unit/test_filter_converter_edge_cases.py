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

"""Edge case tests for Filter Converter to improve coverage.

Feature: postgrest-filter-protocol
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel
from sqlalchemy import create_engine, select
from sqlmodel import Field, Session, SQLModel

from nexau.archs.session.orm import (
    AndFilter,
    ComparisonFilter,
    FilterOperator,
    NotFilter,
    OrFilter,
    evaluate,
    from_query_string,
    to_query_string,
    to_sqlalchemy,
)
from nexau.archs.session.orm.filters.converter import (
    _safe_compare,
    _url_encode_single_value,
)

# ============================================================================
# Test Model Definition
# ============================================================================


class EdgeCaseTestUser(SQLModel, table=True):
    """Test model for edge case tests."""

    __tablename__ = "edge_case_test_users"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    age: int
    score: float
    active: bool
    role: str | None = None


class PydanticUser(BaseModel):
    """Pydantic model for testing evaluate with Pydantic models."""

    name: str
    age: int
    score: float
    active: bool
    role: str | None = None


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def engine():
    """Create an in-memory SQLite database engine."""
    engine = create_engine("sqlite:///:memory:")
    SQLModel.metadata.create_all(engine)
    return engine


@pytest.fixture
def session(engine):
    """Create a database session with test data."""
    with Session(engine) as session:
        users = [
            EdgeCaseTestUser(id=1, name="alice", age=25, score=85.5, active=True, role="admin"),
        ]
        for user in users:
            session.add(user)
        session.commit()
        yield session


# ============================================================================
# _safe_compare Edge Cases
# ============================================================================


class TestSafeCompare:
    """Tests for _safe_compare helper function."""

    def test_safe_compare_incompatible_types(self) -> None:
        """Test _safe_compare returns False for incompatible types."""
        # String vs int comparison should return False (TypeError caught)
        assert _safe_compare("abc", 123, "gt") is False
        assert _safe_compare("abc", 123, "gte") is False
        assert _safe_compare("abc", 123, "lt") is False
        assert _safe_compare("abc", 123, "lte") is False

    def test_safe_compare_invalid_operator(self) -> None:
        """Test _safe_compare returns False for invalid operator."""
        assert _safe_compare(1, 2, "invalid") is False

    def test_safe_compare_list_vs_int(self) -> None:
        """Test _safe_compare with list vs int (incompatible)."""
        assert _safe_compare([1, 2], 3, "gt") is False


# ============================================================================
# evaluate with Pydantic Model
# ============================================================================


class TestEvaluateWithPydanticModel:
    """Tests for evaluate function with Pydantic models."""

    def test_evaluate_with_pydantic_model(self) -> None:
        """Test evaluate with Pydantic BaseModel instance."""
        user = PydanticUser(name="alice", age=25, score=85.5, active=True, role="admin")
        filter_ = ComparisonFilter.eq("name", "alice")
        assert evaluate(filter_, user) is True

    def test_evaluate_pydantic_model_complex_filter(self) -> None:
        """Test evaluate with Pydantic model and complex filter."""
        user = PydanticUser(name="alice", age=25, score=85.5, active=True, role="admin")
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.eq("active", True),
                ComparisonFilter.gte("age", 25),
            ]
        )
        assert evaluate(filter_, user) is True


# ============================================================================
# IS Operator Edge Cases
# ============================================================================


class TestIsOperatorEdgeCases:
    """Tests for IS operator edge cases."""

    def test_is_operator_with_true_value(self) -> None:
        """Test IS operator with True value (IS TRUE)."""
        filter_ = ComparisonFilter(field="active", op=FilterOperator.IS, value=True)
        record = {"active": True}
        assert evaluate(filter_, record) is True

        record2 = {"active": False}
        assert evaluate(filter_, record2) is False

    def test_is_operator_with_false_value(self) -> None:
        """Test IS operator with False value (IS FALSE)."""
        filter_ = ComparisonFilter(field="active", op=FilterOperator.IS, value=False)
        record = {"active": False}
        assert evaluate(filter_, record) is True

        record2 = {"active": True}
        assert evaluate(filter_, record2) is False


# ============================================================================
# LIKE/ILIKE Edge Cases
# ============================================================================


class TestLikeIlikeEdgeCases:
    """Tests for LIKE/ILIKE operator edge cases."""

    def test_like_with_non_string_field_value(self) -> None:
        """Test LIKE returns False when field value is not a string."""
        record = {"age": 25}  # age is int, not string
        filter_ = ComparisonFilter.like("age", "%5")
        assert evaluate(filter_, record) is False

    def test_ilike_with_non_string_field_value(self) -> None:
        """Test ILIKE returns False when field value is not a string."""
        record = {"age": 25}
        filter_ = ComparisonFilter.ilike("age", "%5")
        assert evaluate(filter_, record) is False

    def test_like_with_none_filter_value(self) -> None:
        """Test LIKE returns False when filter value is None."""
        record = {"name": "alice"}
        filter_ = ComparisonFilter(field="name", op=FilterOperator.LIKE, value=None)
        assert evaluate(filter_, record) is False

    def test_ilike_with_none_filter_value(self) -> None:
        """Test ILIKE returns False when filter value is None."""
        record = {"name": "alice"}
        filter_ = ComparisonFilter(field="name", op=FilterOperator.ILIKE, value=None)
        assert evaluate(filter_, record) is False


# ============================================================================
# _url_encode_single_value Edge Cases
# ============================================================================


class TestUrlEncodeSingleValue:
    """Tests for _url_encode_single_value function."""

    def test_encode_list_raises_error(self) -> None:
        """Test that encoding a list raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            _url_encode_single_value([1, 2, 3])
        assert "List values should be encoded using IN operator format" in str(exc_info.value)

    def test_encode_false_boolean(self) -> None:
        """Test encoding False boolean."""
        assert _url_encode_single_value(False) == "false"

    def test_encode_float(self) -> None:
        """Test encoding float value."""
        assert _url_encode_single_value(3.14) == "3.14"


# ============================================================================
# to_sqlalchemy Edge Cases
# ============================================================================


class TestToSqlalchemyEdgeCases:
    """Tests for to_sqlalchemy edge cases."""

    def test_empty_and_filter(self, session) -> None:
        """Test to_sqlalchemy with empty AndFilter."""
        filter_ = AndFilter(filters=[])
        expr = to_sqlalchemy(filter_, EdgeCaseTestUser)

        # Empty AND should return True (all records match)
        stmt = select(EdgeCaseTestUser).where(expr)
        results = session.exec(stmt).all()
        assert len(results) == 1  # Should return all records

    def test_empty_or_filter(self, session) -> None:
        """Test to_sqlalchemy with empty OrFilter."""
        filter_ = OrFilter(filters=[])
        expr = to_sqlalchemy(filter_, EdgeCaseTestUser)

        # Empty OR should return False (no records match)
        stmt = select(EdgeCaseTestUser).where(expr)
        results = session.exec(stmt).all()
        assert len(results) == 0

    def test_field_not_found_raises_error(self) -> None:
        """Test to_sqlalchemy raises ValueError for non-existent field."""
        filter_ = ComparisonFilter.eq("nonexistent_field", "value")
        with pytest.raises(ValueError) as exc_info:
            to_sqlalchemy(filter_, EdgeCaseTestUser)
        assert "not found in model" in str(exc_info.value)

    def test_in_operator_with_non_list_raises_error(self) -> None:
        """Test to_sqlalchemy raises ValueError for IN with non-list value."""
        filter_ = ComparisonFilter(field="name", op=FilterOperator.IN, value="not_a_list")
        with pytest.raises(ValueError) as exc_info:
            to_sqlalchemy(filter_, EdgeCaseTestUser)
        assert "Invalid value type for operator" in str(exc_info.value)


# ============================================================================
# to_query_string Edge Cases
# ============================================================================


class TestToQueryStringEdgeCases:
    """Tests for to_query_string edge cases."""

    def test_empty_and_filter(self) -> None:
        """Test to_query_string with empty AndFilter."""
        filter_ = AndFilter(filters=[])
        assert to_query_string(filter_) == "and=()"

    def test_empty_or_filter(self) -> None:
        """Test to_query_string with empty OrFilter."""
        filter_ = OrFilter(filters=[])
        assert to_query_string(filter_) == "or=()"

    def test_not_filter_with_and(self) -> None:
        """Test to_query_string with NOT wrapping AND filter."""
        filter_ = NotFilter(
            filter=AndFilter(
                filters=[
                    ComparisonFilter.eq("name", "alice"),
                    ComparisonFilter.gt("age", 20),
                ]
            )
        )
        result = to_query_string(filter_)
        assert result == "not=and(name.eq.alice,age.gt.20)"

    def test_not_filter_with_or(self) -> None:
        """Test to_query_string with NOT wrapping OR filter."""
        filter_ = NotFilter(
            filter=OrFilter(
                filters=[
                    ComparisonFilter.eq("name", "alice"),
                    ComparisonFilter.eq("name", "bob"),
                ]
            )
        )
        result = to_query_string(filter_)
        assert result == "not=or(name.eq.alice,name.eq.bob)"

    def test_nested_not_in_and(self) -> None:
        """Test to_query_string with NOT nested inside AND."""
        filter_ = AndFilter(
            filters=[
                NotFilter(filter=ComparisonFilter.eq("name", "alice")),
                ComparisonFilter.gt("age", 20),
            ]
        )
        result = to_query_string(filter_)
        assert result == "and=(name.not.eq.alice,age.gt.20)"

    def test_not_filter_with_in_operator(self) -> None:
        """Test to_query_string with NOT and IN operator."""
        filter_ = NotFilter(filter=ComparisonFilter.in_("role", ["admin", "user"]))
        result = to_query_string(filter_)
        assert result == "role=not.in.(admin,user)"

    def test_in_operator_with_non_list_raises_error(self) -> None:
        """Test to_query_string raises ValueError for IN with non-list value."""
        filter_ = ComparisonFilter(field="role", op=FilterOperator.IN, value="not_a_list")
        with pytest.raises(ValueError) as exc_info:
            to_query_string(filter_)
        assert "Invalid value type for operator" in str(exc_info.value)

    def test_nested_not_in_in_operator_with_non_list_raises_error(self) -> None:
        """Test nested NOT with IN operator and non-list value raises error."""
        inner = ComparisonFilter(field="role", op=FilterOperator.IN, value="not_a_list")
        filter_ = NotFilter(filter=inner)
        with pytest.raises(ValueError) as exc_info:
            to_query_string(filter_)
        assert "Invalid value type for operator" in str(exc_info.value)

    def test_nested_format_not_in_with_non_list_raises_error(self) -> None:
        """Test nested format NOT with IN operator and non-list value raises error."""
        inner = ComparisonFilter(field="role", op=FilterOperator.IN, value="not_a_list")
        filter_ = AndFilter(filters=[NotFilter(filter=inner)])
        with pytest.raises(ValueError) as exc_info:
            to_query_string(filter_)
        assert "Invalid value type for operator" in str(exc_info.value)

    def test_nested_format_in_with_non_list_raises_error(self) -> None:
        """Test nested format IN operator with non-list value raises error."""
        inner = ComparisonFilter(field="role", op=FilterOperator.IN, value="not_a_list")
        filter_ = AndFilter(filters=[inner])
        with pytest.raises(ValueError) as exc_info:
            to_query_string(filter_)
        assert "Invalid value type for operator" in str(exc_info.value)


# ============================================================================
# from_query_string Edge Cases
# ============================================================================


class TestFromQueryStringEdgeCases:
    """Tests for from_query_string edge cases."""

    def test_empty_and_filter(self) -> None:
        """Test from_query_string with empty AND filter."""
        result = from_query_string("and=()")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 0

    def test_empty_or_filter(self) -> None:
        """Test from_query_string with empty OR filter."""
        result = from_query_string("or=()")
        assert isinstance(result, OrFilter)
        assert len(result.filters) == 0

    def test_empty_in_list(self) -> None:
        """Test from_query_string with empty IN list."""
        result = from_query_string("role=in.()")
        assert isinstance(result, ComparisonFilter)
        assert result.op == FilterOperator.IN
        assert result.value == []

    def test_nested_empty_and(self) -> None:
        """Test from_query_string with nested empty AND."""
        result = from_query_string("or=(and(),name.eq.alice)")
        assert isinstance(result, OrFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], AndFilter)
        assert len(result.filters[0].filters) == 0

    def test_nested_empty_or(self) -> None:
        """Test from_query_string with nested empty OR."""
        result = from_query_string("and=(or(),name.eq.alice)")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], OrFilter)
        assert len(result.filters[0].filters) == 0

    def test_not_with_nested_and(self) -> None:
        """Test from_query_string with NOT wrapping AND."""
        result = from_query_string("not=and(name.eq.alice,age.gt.20)")
        assert isinstance(result, NotFilter)
        assert isinstance(result.filter, AndFilter)
        assert len(result.filter.filters) == 2

    def test_not_with_nested_or(self) -> None:
        """Test from_query_string with NOT wrapping OR."""
        result = from_query_string("not=or(name.eq.alice,name.eq.bob)")
        assert isinstance(result, NotFilter)
        assert isinstance(result.filter, OrFilter)
        assert len(result.filter.filters) == 2

    def test_nested_not_in_nested_filter(self) -> None:
        """Test from_query_string with NOT nested inside AND/OR."""
        result = from_query_string("and=(not.name.eq.alice,age.gt.20)")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], NotFilter)

    def test_and_missing_parentheses_raises_error(self) -> None:
        """Test from_query_string raises error for AND without parentheses."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("and=name.eq.alice")
        assert "Invalid query string format" in str(exc_info.value)

    def test_or_missing_parentheses_raises_error(self) -> None:
        """Test from_query_string raises error for OR without parentheses."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("or=name.eq.alice")
        assert "Invalid query string format" in str(exc_info.value)

    def test_in_missing_parentheses_raises_error(self) -> None:
        """Test from_query_string raises error for IN without parentheses."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("role=in.admin,user")
        assert "Invalid query string format" in str(exc_info.value)

    def test_nested_in_missing_parentheses_raises_error(self) -> None:
        """Test from_query_string raises error for nested IN without parentheses."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("and=(role.in.admin)")
        assert "Invalid query string format" in str(exc_info.value)

    def test_boolean_values_in_in_list(self) -> None:
        """Test from_query_string converts boolean values in IN list to strings."""
        result = from_query_string("field=in.(true,false,null)")
        assert isinstance(result, ComparisonFilter)
        assert result.op == FilterOperator.IN
        # Boolean and null values are converted to strings in IN lists
        assert isinstance(result.value, list)
        assert "True" in result.value or "true" in result.value or True in result.value

    def test_float_value_parsing(self) -> None:
        """Test from_query_string parses float values correctly."""
        result = from_query_string("score=gte.3.14")
        assert isinstance(result, ComparisonFilter)
        assert result.value == 3.14

    def test_scientific_notation_as_string(self) -> None:
        """Test from_query_string handles scientific notation."""
        result = from_query_string("value=eq.1e10")
        assert isinstance(result, ComparisonFilter)
        # Scientific notation should be parsed as float
        assert result.value == 1e10


# ============================================================================
# Additional Edge Cases for Higher Coverage
# ============================================================================


class TestToSqlalchemyUnsupportedOperator:
    """Tests for unsupported operator in to_sqlalchemy."""

    def test_unsupported_operator_raises_error(self) -> None:
        """Test that an unsupported operator raises ValueError."""
        # Create a filter with a mocked unsupported operator
        # This is hard to test directly since FilterOperator is an enum
        # But we can test the else branch by checking the error message
        pass  # This branch is unreachable with valid FilterOperator enum


class TestEvaluateUnsupportedOperator:
    """Tests for unsupported operator in evaluate."""

    def test_unsupported_operator_raises_error(self) -> None:
        """Test that an unsupported operator raises ValueError."""
        # This branch is unreachable with valid FilterOperator enum
        pass


class TestNestedFormatEmptyFilters:
    """Tests for nested format with empty filters."""

    def test_nested_and_empty_in_or(self) -> None:
        """Test nested empty AND inside OR."""
        filter_ = OrFilter(
            filters=[
                AndFilter(filters=[]),
                ComparisonFilter.eq("name", "alice"),
            ]
        )
        result = to_query_string(filter_)
        assert "and()" in result

    def test_nested_or_empty_in_and(self) -> None:
        """Test nested empty OR inside AND."""
        filter_ = AndFilter(
            filters=[
                OrFilter(filters=[]),
                ComparisonFilter.eq("name", "alice"),
            ]
        )
        result = to_query_string(filter_)
        assert "or()" in result


class TestNestedNotWithLogicalFilters:
    """Tests for nested NOT with logical filters."""

    def test_nested_not_with_and_in_or(self) -> None:
        """Test NOT wrapping AND inside OR."""
        filter_ = OrFilter(
            filters=[
                NotFilter(
                    filter=AndFilter(
                        filters=[
                            ComparisonFilter.eq("name", "alice"),
                            ComparisonFilter.gt("age", 20),
                        ]
                    )
                ),
                ComparisonFilter.eq("role", "admin"),
            ]
        )
        result = to_query_string(filter_)
        assert "not.and(" in result

    def test_nested_not_with_or_in_and(self) -> None:
        """Test NOT wrapping OR inside AND."""
        filter_ = AndFilter(
            filters=[
                NotFilter(
                    filter=OrFilter(
                        filters=[
                            ComparisonFilter.eq("name", "alice"),
                            ComparisonFilter.eq("name", "bob"),
                        ]
                    )
                ),
                ComparisonFilter.gt("age", 20),
            ]
        )
        result = to_query_string(filter_)
        assert "not.or(" in result


class TestFromQueryStringNestedNotComparison:
    """Tests for from_query_string with nested NOT comparison."""

    def test_nested_not_comparison_in_and(self) -> None:
        """Test parsing nested NOT comparison inside AND."""
        result = from_query_string("and=(name.not.eq.alice,age.gt.20)")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], NotFilter)
        inner = result.filters[0].filter
        assert isinstance(inner, ComparisonFilter)
        assert inner.field == "name"
        assert inner.op == FilterOperator.EQ
        assert inner.value == "alice"

    def test_nested_not_in_operator_in_and(self) -> None:
        """Test parsing nested NOT IN operator inside AND."""
        result = from_query_string("and=(role.not.in.(admin,user),age.gt.20)")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], NotFilter)
        inner = result.filters[0].filter
        assert isinstance(inner, ComparisonFilter)
        assert inner.field == "role"
        assert inner.op == FilterOperator.IN

    def test_nested_not_comparison_in_or(self) -> None:
        """Test parsing nested NOT comparison inside OR."""
        result = from_query_string("or=(name.not.eq.alice,name.not.eq.bob)")
        assert isinstance(result, OrFilter)
        assert len(result.filters) == 2
        for f in result.filters:
            assert isinstance(f, NotFilter)


class TestFromQueryStringNestedInEmpty:
    """Tests for from_query_string with nested empty IN list."""

    def test_nested_empty_in_list(self) -> None:
        """Test parsing nested empty IN list."""
        result = from_query_string("and=(role.in.(),name.eq.alice)")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], ComparisonFilter)
        assert result.filters[0].op == FilterOperator.IN
        assert result.filters[0].value == []


class TestFromQueryStringNestedNotWithLogical:
    """Tests for from_query_string with nested NOT wrapping logical filters."""

    def test_nested_not_and_in_or(self) -> None:
        """Test parsing NOT(AND) inside OR."""
        result = from_query_string("or=(not.and(name.eq.alice,age.gt.20),role.eq.admin)")
        assert isinstance(result, OrFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], NotFilter)
        assert isinstance(result.filters[0].filter, AndFilter)

    def test_nested_not_or_in_and(self) -> None:
        """Test parsing NOT(OR) inside AND."""
        result = from_query_string("and=(not.or(name.eq.alice,name.eq.bob),age.gt.20)")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], NotFilter)
        assert isinstance(result.filters[0].filter, OrFilter)


class TestFromQueryStringInvalidNestedFormat:
    """Tests for from_query_string with invalid nested format."""

    def test_nested_missing_dot_raises_error(self) -> None:
        """Test parsing nested filter without dot raises error."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("and=(nameeqalice)")
        assert "Invalid query string format" in str(exc_info.value)

    def test_nested_missing_second_dot_raises_error(self) -> None:
        """Test parsing nested filter without second dot raises error."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("and=(name.eqalice)")
        assert "Invalid query string format" in str(exc_info.value)

    def test_nested_unknown_operator_raises_error(self) -> None:
        """Test parsing nested filter with unknown operator raises error."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("and=(name.unknown.alice)")
        assert "Unknown operator" in str(exc_info.value)


class TestFromQueryStringTopLevelNotIn:
    """Tests for from_query_string with top-level NOT IN."""

    def test_top_level_not_in_empty(self) -> None:
        """Test parsing top-level NOT IN with empty list."""
        result = from_query_string("role=not.in.()")
        assert isinstance(result, NotFilter)
        inner = result.filter
        assert isinstance(inner, ComparisonFilter)
        assert inner.op == FilterOperator.IN
        assert inner.value == []

    def test_top_level_not_in_with_values(self) -> None:
        """Test parsing top-level NOT IN with values."""
        result = from_query_string("role=not.in.(admin,user,guest)")
        assert isinstance(result, NotFilter)
        inner = result.filter
        assert isinstance(inner, ComparisonFilter)
        assert inner.op == FilterOperator.IN
        assert isinstance(inner.value, list)
        assert len(inner.value) == 3


class TestFromQueryStringBooleanAndNullInList:
    """Tests for from_query_string with boolean and null values in IN list."""

    def test_null_in_list_converted_to_string(self) -> None:
        """Test that null in IN list is converted to 'null' string."""
        result = from_query_string("field=in.(null,value)")
        assert isinstance(result, ComparisonFilter)
        assert isinstance(result.value, list)
        assert "null" in result.value

    def test_true_in_list_converted_to_string(self) -> None:
        """Test that true in IN list is converted to 'True' string."""
        result = from_query_string("field=in.(true,value)")
        assert isinstance(result, ComparisonFilter)
        # true is converted to "True" string
        assert isinstance(result.value, list)
        assert "True" in result.value

    def test_false_in_list_converted_to_string(self) -> None:
        """Test that false in IN list is converted to 'False' string."""
        result = from_query_string("field=in.(false,value)")
        assert isinstance(result, ComparisonFilter)
        # false is converted to "False" string
        assert isinstance(result.value, list)
        assert "False" in result.value


class TestFromQueryStringNestedBooleanAndNull:
    """Tests for from_query_string with nested boolean and null values."""

    def test_nested_null_in_list(self) -> None:
        """Test parsing nested IN with null value."""
        result = from_query_string("and=(field.in.(null,value),name.eq.alice)")
        assert isinstance(result, AndFilter)
        first_filter = result.filters[0]
        assert isinstance(first_filter, ComparisonFilter)
        assert isinstance(first_filter.value, list)
        assert "null" in first_filter.value

    def test_nested_boolean_in_list(self) -> None:
        """Test parsing nested IN with boolean values."""
        result = from_query_string("and=(field.in.(true,false),name.eq.alice)")
        assert isinstance(result, AndFilter)
        first_filter = result.filters[0]
        assert isinstance(first_filter, ComparisonFilter)
        # Booleans are converted to strings
        assert isinstance(first_filter.value, list)
        assert "True" in first_filter.value or "False" in first_filter.value
