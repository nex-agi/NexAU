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

"""Tests for the Filter Converter to_sqlalchemy function.

Feature: postgrest-filter-protocol
"""

from __future__ import annotations

import pytest
from sqlalchemy import create_engine, select
from sqlmodel import Field, Session, SQLModel

from nexau.archs.session.orm import (
    AndFilter,
    ComparisonFilter,
    FilterOperator,
    NotFilter,
    OrFilter,
    from_query_string,
    to_query_string,
    to_sqlalchemy,
)

# ============================================================================
# Test Model Definition
# ============================================================================


class FilterTestUser(SQLModel, table=True):
    """Test model for filter converter tests."""

    __tablename__ = "filter_converter_test_users"

    id: int | None = Field(default=None, primary_key=True)
    name: str
    age: int
    email: str
    score: float
    active: bool
    role: str | None = None


# ============================================================================
# Test Fixtures
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
        # Insert test data
        users = [
            FilterTestUser(id=1, name="alice", age=25, email="alice@example.com", score=85.5, active=True, role="admin"),
            FilterTestUser(id=2, name="bob", age=30, email="bob@test.com", score=72.0, active=True, role="user"),
            FilterTestUser(id=3, name="charlie", age=35, email="charlie@example.com", score=90.5, active=False, role="user"),
            FilterTestUser(id=4, name="diana", age=28, email="diana@test.org", score=65.0, active=True, role=None),
            FilterTestUser(id=5, name="Alice Smith", age=22, email="asmith@example.com", score=88.0, active=True, role="moderator"),
        ]
        for user in users:
            session.add(user)
        session.commit()
        yield session


# ============================================================================
# ComparisonFilter Tests - Requirement 2.2-2.8
# ============================================================================


class TestComparisonFilterToSqlalchemy:
    """Tests for ComparisonFilter to SQLAlchemy conversion."""

    def test_eq_operator(self, session) -> None:
        """Test eq operator conversion (Requirement 2.2)."""
        filter_ = ComparisonFilter.eq("name", "alice")
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 1
        assert result[0].name == "alice"

    def test_neq_operator(self, session) -> None:
        """Test neq operator conversion (Requirement 2.3)."""
        filter_ = ComparisonFilter.neq("name", "alice")
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 4
        assert all(u.name != "alice" for u in result)

    def test_gt_operator(self, session) -> None:
        """Test gt operator conversion (Requirement 2.4)."""
        filter_ = ComparisonFilter.gt("age", 28)
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 2
        assert all(u.age > 28 for u in result)

    def test_gte_operator(self, session) -> None:
        """Test gte operator conversion (Requirement 2.4)."""
        filter_ = ComparisonFilter.gte("age", 28)
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 3
        assert all(u.age >= 28 for u in result)

    def test_lt_operator(self, session) -> None:
        """Test lt operator conversion (Requirement 2.4)."""
        filter_ = ComparisonFilter.lt("age", 28)
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 2
        assert all(u.age < 28 for u in result)

    def test_lte_operator(self, session) -> None:
        """Test lte operator conversion (Requirement 2.4)."""
        filter_ = ComparisonFilter.lte("age", 28)
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 3
        assert all(u.age <= 28 for u in result)

    def test_like_operator(self, session) -> None:
        """Test like operator conversion (Requirement 2.5)."""
        filter_ = ComparisonFilter.like("email", "%@example.com")
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 3
        assert all("@example.com" in u.email for u in result)

    def test_ilike_operator(self, session) -> None:
        """Test ilike operator conversion (Requirement 2.6)."""
        filter_ = ComparisonFilter.ilike("name", "alice%")
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        # Should match both "alice" and "Alice Smith" (case-insensitive)
        assert len(result) == 2

    def test_in_operator(self, session) -> None:
        """Test in operator conversion (Requirement 2.7)."""
        filter_ = ComparisonFilter.in_("role", ["admin", "moderator"])
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 2
        assert all(u.role in ["admin", "moderator"] for u in result)

    def test_in_operator_empty_list(self, session) -> None:
        """Test in operator with empty list."""
        filter_ = ComparisonFilter.in_("role", [])
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 0

    def test_is_null_operator(self, session) -> None:
        """Test is operator for NULL check (Requirement 2.8)."""
        filter_ = ComparisonFilter.is_null("role")
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 1
        assert result[0].name == "diana"
        assert result[0].role is None

    def test_eq_with_float(self, session) -> None:
        """Test eq operator with float value."""
        filter_ = ComparisonFilter.eq("score", 85.5)
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 1
        assert result[0].name == "alice"

    def test_eq_with_bool(self, session) -> None:
        """Test eq operator with boolean value."""
        filter_ = ComparisonFilter.eq("active", False)
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 1
        assert result[0].name == "charlie"


# ============================================================================
# LogicalFilter Tests - Requirement 2.9-2.11
# ============================================================================


class TestLogicalFilterToSqlalchemy:
    """Tests for LogicalFilter to SQLAlchemy conversion."""

    def test_and_filter(self, session) -> None:
        """Test AND filter conversion (Requirement 2.9)."""
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.eq("active", True),
                ComparisonFilter.gt("age", 25),
            ]
        )
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 2
        assert all(u.active is True and u.age > 25 for u in result)

    def test_and_filter_empty(self, session) -> None:
        """Test AND filter with empty filters list."""
        filter_ = AndFilter(filters=[])
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        # Empty AND should return all records (True)
        assert len(result) == 5

    def test_or_filter(self, session) -> None:
        """Test OR filter conversion (Requirement 2.10)."""
        filter_ = OrFilter(
            filters=[
                ComparisonFilter.eq("role", "admin"),
                ComparisonFilter.eq("role", "moderator"),
            ]
        )
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 2
        assert all(u.role in ["admin", "moderator"] for u in result)

    def test_or_filter_empty(self, session) -> None:
        """Test OR filter with empty filters list."""
        filter_ = OrFilter(filters=[])
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        # Empty OR should return no records (False)
        assert len(result) == 0

    def test_not_filter(self, session) -> None:
        """Test NOT filter conversion (Requirement 2.11)."""
        filter_ = NotFilter(filter=ComparisonFilter.eq("active", True))
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 1
        assert result[0].active is False

    def test_nested_logical_filters(self, session) -> None:
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
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        # bob (active, age 30), diana (active, age 28), alice (admin)
        assert len(result) == 3

    def test_deeply_nested_filters(self, session) -> None:
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
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        # Excludes: bob (user), charlie (user, also inactive and age > 30)
        # Note: diana (role=None) is also excluded due to SQL NULL semantics
        # In SQL, NULL = 'user' returns NULL, and NOT(NULL OR False) = NOT(NULL) = NULL
        # Includes: alice (admin), Alice Smith (moderator)
        assert len(result) == 2
        names = {u.name for u in result}
        assert names == {"alice", "Alice Smith"}


# ============================================================================
# Error Handling Tests - Requirement 2.12
# ============================================================================


class TestFilterConverterErrors:
    """Tests for error handling in filter converter."""

    def test_invalid_field_name_raises_value_error(self) -> None:
        """Test that invalid field name raises ValueError (Requirement 2.12)."""
        filter_ = ComparisonFilter.eq("nonexistent_field", "value")

        with pytest.raises(ValueError) as exc_info:
            to_sqlalchemy(filter_, FilterTestUser)

        assert "Field 'nonexistent_field' not found in model FilterTestUser" in str(exc_info.value)

    def test_invalid_field_in_and_filter(self) -> None:
        """Test that invalid field in AndFilter raises ValueError."""
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.eq("name", "alice"),
                ComparisonFilter.eq("invalid_field", "value"),
            ]
        )

        with pytest.raises(ValueError) as exc_info:
            to_sqlalchemy(filter_, FilterTestUser)

        assert "Field 'invalid_field' not found in model FilterTestUser" in str(exc_info.value)

    def test_invalid_field_in_or_filter(self) -> None:
        """Test that invalid field in OrFilter raises ValueError."""
        filter_ = OrFilter(
            filters=[
                ComparisonFilter.eq("name", "alice"),
                ComparisonFilter.eq("bad_field", "value"),
            ]
        )

        with pytest.raises(ValueError) as exc_info:
            to_sqlalchemy(filter_, FilterTestUser)

        assert "Field 'bad_field' not found in model FilterTestUser" in str(exc_info.value)

    def test_invalid_field_in_not_filter(self) -> None:
        """Test that invalid field in NotFilter raises ValueError."""
        filter_ = NotFilter(filter=ComparisonFilter.eq("missing_field", "value"))

        with pytest.raises(ValueError) as exc_info:
            to_sqlalchemy(filter_, FilterTestUser)

        assert "Field 'missing_field' not found in model FilterTestUser" in str(exc_info.value)

    def test_in_operator_with_non_list_value(self) -> None:
        """Test that IN operator with non-list value raises ValueError."""
        # Manually create a filter with invalid value type for IN
        filter_ = ComparisonFilter(field="role", op=FilterOperator.IN, value="not_a_list")

        with pytest.raises(ValueError) as exc_info:
            to_sqlalchemy(filter_, FilterTestUser)

        assert "Invalid value type for operator" in str(exc_info.value)


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestFilterConverterEdgeCases:
    """Tests for edge cases in filter converter."""

    def test_single_filter_in_and(self, session) -> None:
        """Test AndFilter with single filter."""
        filter_ = AndFilter(filters=[ComparisonFilter.eq("name", "alice")])
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 1
        assert result[0].name == "alice"

    def test_single_filter_in_or(self, session) -> None:
        """Test OrFilter with single filter."""
        filter_ = OrFilter(filters=[ComparisonFilter.eq("name", "bob")])
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 1
        assert result[0].name == "bob"

    def test_double_negation(self, session) -> None:
        """Test double negation (NOT NOT)."""
        filter_ = NotFilter(filter=NotFilter(filter=ComparisonFilter.eq("name", "alice")))
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 1
        assert result[0].name == "alice"

    def test_like_with_percent_wildcard(self, session) -> None:
        """Test LIKE with % wildcard at different positions."""
        # Prefix match - Note: SQLite LIKE is case-insensitive by default
        filter1 = ComparisonFilter.like("name", "a%")
        expr1 = to_sqlalchemy(filter1, FilterTestUser)
        result1 = session.exec(select(FilterTestUser).where(expr1)).scalars().all()
        assert len(result1) == 2  # "alice" and "Alice Smith" (SQLite LIKE is case-insensitive)

        # Suffix match
        filter2 = ComparisonFilter.like("email", "%@test.com")
        expr2 = to_sqlalchemy(filter2, FilterTestUser)
        result2 = session.exec(select(FilterTestUser).where(expr2)).scalars().all()
        assert len(result2) == 1  # bob@test.com

        # Contains match
        filter3 = ComparisonFilter.like("email", "%example%")
        expr3 = to_sqlalchemy(filter3, FilterTestUser)
        result3 = session.exec(select(FilterTestUser).where(expr3)).scalars().all()
        assert len(result3) == 3

    def test_comparison_with_integer_id(self, session) -> None:
        """Test comparison with integer ID field."""
        filter_ = ComparisonFilter.in_("id", [1, 3, 5])
        expr = to_sqlalchemy(filter_, FilterTestUser)

        result = session.exec(select(FilterTestUser).where(expr)).scalars().all()
        assert len(result) == 3
        assert {u.id for u in result} == {1, 3, 5}


# ============================================================================
# to_query_string Tests - Requirement 4.1-4.7
# ============================================================================


class TestToQueryStringComparison:
    """Tests for ComparisonFilter to query string conversion."""

    def test_eq_operator(self) -> None:
        """Test eq operator serialization (Requirement 4.2)."""
        filter_ = ComparisonFilter.eq("name", "alice")
        result = to_query_string(filter_)
        assert result == "name=eq.alice"

    def test_neq_operator(self) -> None:
        """Test neq operator serialization (Requirement 4.2)."""
        filter_ = ComparisonFilter.neq("status", "deleted")
        result = to_query_string(filter_)
        assert result == "status=neq.deleted"

    def test_gt_operator(self) -> None:
        """Test gt operator serialization (Requirement 4.2)."""
        filter_ = ComparisonFilter.gt("age", 18)
        result = to_query_string(filter_)
        assert result == "age=gt.18"

    def test_gte_operator(self) -> None:
        """Test gte operator serialization (Requirement 4.2)."""
        filter_ = ComparisonFilter.gte("score", 85.5)
        result = to_query_string(filter_)
        assert result == "score=gte.85.5"

    def test_lt_operator(self) -> None:
        """Test lt operator serialization (Requirement 4.2)."""
        filter_ = ComparisonFilter.lt("count", 100)
        result = to_query_string(filter_)
        assert result == "count=lt.100"

    def test_lte_operator(self) -> None:
        """Test lte operator serialization (Requirement 4.2)."""
        filter_ = ComparisonFilter.lte("price", 99.99)
        result = to_query_string(filter_)
        assert result == "price=lte.99.99"

    def test_like_operator(self) -> None:
        """Test like operator serialization (Requirement 4.2)."""
        filter_ = ComparisonFilter.like("email", "%@example.com")
        result = to_query_string(filter_)
        # % should be URL encoded as %25
        assert result == "email=like.%25%40example.com"

    def test_ilike_operator(self) -> None:
        """Test ilike operator serialization (Requirement 4.2)."""
        filter_ = ComparisonFilter.ilike("name", "alice%")
        result = to_query_string(filter_)
        assert result == "name=ilike.alice%25"

    def test_in_operator(self) -> None:
        """Test in operator serialization (Requirement 4.3)."""
        filter_ = ComparisonFilter.in_("status", ["active", "pending"])
        result = to_query_string(filter_)
        assert result == "status=in.(active,pending)"

    def test_in_operator_with_integers(self) -> None:
        """Test in operator with integer values (Requirement 4.3)."""
        filter_ = ComparisonFilter.in_("id", [1, 2, 3])
        result = to_query_string(filter_)
        assert result == "id=in.(1,2,3)"

    def test_in_operator_empty_list(self) -> None:
        """Test in operator with empty list (Requirement 4.3)."""
        filter_ = ComparisonFilter.in_("role", [])
        result = to_query_string(filter_)
        assert result == "role=in.()"

    def test_is_null_operator(self) -> None:
        """Test is operator for NULL (Requirement 4.2)."""
        filter_ = ComparisonFilter.is_null("role")
        result = to_query_string(filter_)
        assert result == "role=is.null"

    def test_eq_with_boolean_true(self) -> None:
        """Test eq operator with boolean true value."""
        filter_ = ComparisonFilter.eq("active", True)
        result = to_query_string(filter_)
        assert result == "active=eq.true"

    def test_eq_with_boolean_false(self) -> None:
        """Test eq operator with boolean false value."""
        filter_ = ComparisonFilter.eq("active", False)
        result = to_query_string(filter_)
        assert result == "active=eq.false"


class TestToQueryStringLogical:
    """Tests for LogicalFilter to query string conversion."""

    def test_and_filter(self) -> None:
        """Test AND filter serialization (Requirement 4.4)."""
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.gte("age", 18),
                ComparisonFilter.eq("status", "active"),
            ]
        )
        result = to_query_string(filter_)
        assert result == "and=(age.gte.18,status.eq.active)"

    def test_and_filter_empty(self) -> None:
        """Test AND filter with empty filters list (Requirement 4.4)."""
        filter_ = AndFilter(filters=[])
        result = to_query_string(filter_)
        assert result == "and=()"

    def test_or_filter(self) -> None:
        """Test OR filter serialization (Requirement 4.5)."""
        filter_ = OrFilter(
            filters=[
                ComparisonFilter.eq("role", "admin"),
                ComparisonFilter.eq("role", "moderator"),
            ]
        )
        result = to_query_string(filter_)
        assert result == "or=(role.eq.admin,role.eq.moderator)"

    def test_or_filter_empty(self) -> None:
        """Test OR filter with empty filters list (Requirement 4.5)."""
        filter_ = OrFilter(filters=[])
        result = to_query_string(filter_)
        assert result == "or=()"

    def test_not_filter_with_comparison(self) -> None:
        """Test NOT filter with ComparisonFilter (Requirement 4.6)."""
        filter_ = NotFilter(filter=ComparisonFilter.eq("status", "deleted"))
        result = to_query_string(filter_)
        assert result == "status=not.eq.deleted"

    def test_not_filter_with_in_operator(self) -> None:
        """Test NOT filter with IN operator (Requirement 4.6)."""
        filter_ = NotFilter(filter=ComparisonFilter.in_("role", ["banned", "suspended"]))
        result = to_query_string(filter_)
        assert result == "role=not.in.(banned,suspended)"

    def test_not_filter_with_and(self) -> None:
        """Test NOT filter with AndFilter (Requirement 4.6)."""
        filter_ = NotFilter(
            filter=AndFilter(
                filters=[
                    ComparisonFilter.eq("active", False),
                    ComparisonFilter.gt("age", 30),
                ]
            )
        )
        result = to_query_string(filter_)
        assert result == "not=and(active.eq.false,age.gt.30)"

    def test_not_filter_with_or(self) -> None:
        """Test NOT filter with OrFilter (Requirement 4.6)."""
        filter_ = NotFilter(
            filter=OrFilter(
                filters=[
                    ComparisonFilter.eq("role", "user"),
                    ComparisonFilter.eq("role", "guest"),
                ]
            )
        )
        result = to_query_string(filter_)
        assert result == "not=or(role.eq.user,role.eq.guest)"


class TestToQueryStringNested:
    """Tests for nested filter serialization."""

    def test_nested_and_in_or(self) -> None:
        """Test nested AND inside OR."""
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
        result = to_query_string(filter_)
        assert result == "or=(and(active.eq.true,age.gt.25),role.eq.admin)"

    def test_nested_or_in_and(self) -> None:
        """Test nested OR inside AND."""
        filter_ = AndFilter(
            filters=[
                OrFilter(
                    filters=[
                        ComparisonFilter.eq("role", "admin"),
                        ComparisonFilter.eq("role", "moderator"),
                    ]
                ),
                ComparisonFilter.eq("active", True),
            ]
        )
        result = to_query_string(filter_)
        assert result == "and=(or(role.eq.admin,role.eq.moderator),active.eq.true)"

    def test_deeply_nested_filters(self) -> None:
        """Test deeply nested filters."""
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
        result = to_query_string(filter_)
        assert result == "not=or(role.eq.user,and(active.eq.false,age.gt.30))"


class TestToQueryStringUrlEncoding:
    """Tests for URL encoding of special characters (Requirement 4.7)."""

    def test_url_encode_space(self) -> None:
        """Test URL encoding of space character."""
        filter_ = ComparisonFilter.eq("name", "alice smith")
        result = to_query_string(filter_)
        assert result == "name=eq.alice%20smith"

    def test_url_encode_ampersand(self) -> None:
        """Test URL encoding of ampersand."""
        filter_ = ComparisonFilter.eq("company", "A&B Corp")
        result = to_query_string(filter_)
        assert result == "company=eq.A%26B%20Corp"

    def test_url_encode_equals(self) -> None:
        """Test URL encoding of equals sign."""
        filter_ = ComparisonFilter.eq("equation", "x=y")
        result = to_query_string(filter_)
        assert result == "equation=eq.x%3Dy"

    def test_url_encode_parentheses(self) -> None:
        """Test URL encoding of parentheses."""
        filter_ = ComparisonFilter.eq("note", "(important)")
        result = to_query_string(filter_)
        assert result == "note=eq.%28important%29"

    def test_url_encode_comma(self) -> None:
        """Test URL encoding of comma."""
        filter_ = ComparisonFilter.eq("list", "a,b,c")
        result = to_query_string(filter_)
        assert result == "list=eq.a%2Cb%2Cc"

    def test_url_encode_in_list_values(self) -> None:
        """Test URL encoding in IN operator list values."""
        filter_ = ComparisonFilter.in_("tag", ["hello world", "foo&bar"])
        result = to_query_string(filter_)
        assert result == "tag=in.(hello%20world,foo%26bar)"

    def test_url_encode_unicode(self) -> None:
        """Test URL encoding of unicode characters."""
        filter_ = ComparisonFilter.eq("name", "日本語")
        result = to_query_string(filter_)
        # Unicode characters should be percent-encoded
        assert "name=eq." in result
        assert "%E6%97%A5%E6%9C%AC%E8%AA%9E" in result

    def test_url_encode_special_chars_in_nested(self) -> None:
        """Test URL encoding in nested filters."""
        filter_ = AndFilter(
            filters=[
                ComparisonFilter.eq("name", "alice smith"),
                ComparisonFilter.eq("company", "A&B"),
            ]
        )
        result = to_query_string(filter_)
        assert result == "and=(name.eq.alice%20smith,company.eq.A%26B)"


# ============================================================================
# from_query_string Tests - Requirement 4.8
# ============================================================================


class TestFromQueryStringComparison:
    """Tests for ComparisonFilter parsing from query string."""

    def test_eq_operator(self) -> None:
        """Test eq operator parsing (Requirement 4.8)."""
        result = from_query_string("name=eq.alice")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "name"
        assert result.op == FilterOperator.EQ
        assert result.value == "alice"

    def test_neq_operator(self) -> None:
        """Test neq operator parsing (Requirement 4.8)."""
        result = from_query_string("status=neq.deleted")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "status"
        assert result.op == FilterOperator.NEQ
        assert result.value == "deleted"

    def test_gt_operator(self) -> None:
        """Test gt operator parsing (Requirement 4.8)."""
        result = from_query_string("age=gt.18")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "age"
        assert result.op == FilterOperator.GT
        assert result.value == 18

    def test_gte_operator(self) -> None:
        """Test gte operator parsing (Requirement 4.8)."""
        result = from_query_string("score=gte.85.5")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "score"
        assert result.op == FilterOperator.GTE
        assert result.value == 85.5

    def test_lt_operator(self) -> None:
        """Test lt operator parsing (Requirement 4.8)."""
        result = from_query_string("count=lt.100")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "count"
        assert result.op == FilterOperator.LT
        assert result.value == 100

    def test_lte_operator(self) -> None:
        """Test lte operator parsing (Requirement 4.8)."""
        result = from_query_string("price=lte.99.99")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "price"
        assert result.op == FilterOperator.LTE
        assert result.value == 99.99

    def test_like_operator(self) -> None:
        """Test like operator parsing with URL decoding (Requirement 4.8)."""
        result = from_query_string("email=like.%25%40example.com")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "email"
        assert result.op == FilterOperator.LIKE
        assert result.value == "%@example.com"

    def test_ilike_operator(self) -> None:
        """Test ilike operator parsing (Requirement 4.8)."""
        result = from_query_string("name=ilike.alice%25")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "name"
        assert result.op == FilterOperator.ILIKE
        assert result.value == "alice%"

    def test_in_operator(self) -> None:
        """Test in operator parsing (Requirement 4.8)."""
        result = from_query_string("status=in.(active,pending)")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "status"
        assert result.op == FilterOperator.IN
        assert result.value == ["active", "pending"]

    def test_in_operator_with_integers(self) -> None:
        """Test in operator with integer values (Requirement 4.8)."""
        result = from_query_string("id=in.(1,2,3)")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "id"
        assert result.op == FilterOperator.IN
        assert result.value == [1, 2, 3]

    def test_in_operator_empty_list(self) -> None:
        """Test in operator with empty list (Requirement 4.8)."""
        result = from_query_string("role=in.()")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "role"
        assert result.op == FilterOperator.IN
        assert result.value == []

    def test_is_null_operator(self) -> None:
        """Test is operator for NULL (Requirement 4.8)."""
        result = from_query_string("role=is.null")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "role"
        assert result.op == FilterOperator.IS
        assert result.value is None

    def test_eq_with_boolean_true(self) -> None:
        """Test eq operator with boolean true value."""
        result = from_query_string("active=eq.true")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "active"
        assert result.op == FilterOperator.EQ
        assert result.value is True

    def test_eq_with_boolean_false(self) -> None:
        """Test eq operator with boolean false value."""
        result = from_query_string("active=eq.false")
        assert isinstance(result, ComparisonFilter)
        assert result.field == "active"
        assert result.op == FilterOperator.EQ
        assert result.value is False


class TestFromQueryStringLogical:
    """Tests for LogicalFilter parsing from query string."""

    def test_and_filter(self) -> None:
        """Test AND filter parsing (Requirement 4.8)."""
        result = from_query_string("and=(age.gte.18,status.eq.active)")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], ComparisonFilter)
        assert result.filters[0].field == "age"
        assert result.filters[0].op == FilterOperator.GTE
        assert result.filters[0].value == 18
        assert isinstance(result.filters[1], ComparisonFilter)
        assert result.filters[1].field == "status"
        assert result.filters[1].op == FilterOperator.EQ
        assert result.filters[1].value == "active"

    def test_and_filter_empty(self) -> None:
        """Test AND filter with empty filters list (Requirement 4.8)."""
        result = from_query_string("and=()")
        assert isinstance(result, AndFilter)
        assert result.filters == []

    def test_or_filter(self) -> None:
        """Test OR filter parsing (Requirement 4.8)."""
        result = from_query_string("or=(role.eq.admin,role.eq.moderator)")
        assert isinstance(result, OrFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], ComparisonFilter)
        assert result.filters[0].field == "role"
        assert result.filters[0].value == "admin"
        assert isinstance(result.filters[1], ComparisonFilter)
        assert result.filters[1].field == "role"
        assert result.filters[1].value == "moderator"

    def test_or_filter_empty(self) -> None:
        """Test OR filter with empty filters list (Requirement 4.8)."""
        result = from_query_string("or=()")
        assert isinstance(result, OrFilter)
        assert result.filters == []

    def test_not_filter_with_comparison(self) -> None:
        """Test NOT filter with ComparisonFilter (Requirement 4.8)."""
        result = from_query_string("status=not.eq.deleted")
        assert isinstance(result, NotFilter)
        assert isinstance(result.filter, ComparisonFilter)
        assert result.filter.field == "status"
        assert result.filter.op == FilterOperator.EQ
        assert result.filter.value == "deleted"

    def test_not_filter_with_in_operator(self) -> None:
        """Test NOT filter with IN operator (Requirement 4.8)."""
        result = from_query_string("role=not.in.(banned,suspended)")
        assert isinstance(result, NotFilter)
        assert isinstance(result.filter, ComparisonFilter)
        assert result.filter.field == "role"
        assert result.filter.op == FilterOperator.IN
        assert result.filter.value == ["banned", "suspended"]

    def test_not_filter_with_and(self) -> None:
        """Test NOT filter with AndFilter (Requirement 4.8)."""
        result = from_query_string("not=and(active.eq.false,age.gt.30)")
        assert isinstance(result, NotFilter)
        assert isinstance(result.filter, AndFilter)
        assert len(result.filter.filters) == 2

    def test_not_filter_with_or(self) -> None:
        """Test NOT filter with OrFilter (Requirement 4.8)."""
        result = from_query_string("not=or(role.eq.user,role.eq.guest)")
        assert isinstance(result, NotFilter)
        assert isinstance(result.filter, OrFilter)
        assert len(result.filter.filters) == 2


class TestFromQueryStringNested:
    """Tests for nested filter parsing."""

    def test_nested_and_in_or(self) -> None:
        """Test nested AND inside OR."""
        result = from_query_string("or=(and(active.eq.true,age.gt.25),role.eq.admin)")
        assert isinstance(result, OrFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], AndFilter)
        assert len(result.filters[0].filters) == 2
        assert isinstance(result.filters[1], ComparisonFilter)

    def test_nested_or_in_and(self) -> None:
        """Test nested OR inside AND."""
        result = from_query_string("and=(or(role.eq.admin,role.eq.moderator),active.eq.true)")
        assert isinstance(result, AndFilter)
        assert len(result.filters) == 2
        assert isinstance(result.filters[0], OrFilter)
        assert len(result.filters[0].filters) == 2
        assert isinstance(result.filters[1], ComparisonFilter)

    def test_deeply_nested_filters(self) -> None:
        """Test deeply nested filters."""
        result = from_query_string("not=or(role.eq.user,and(active.eq.false,age.gt.30))")
        assert isinstance(result, NotFilter)
        assert isinstance(result.filter, OrFilter)
        assert len(result.filter.filters) == 2
        assert isinstance(result.filter.filters[0], ComparisonFilter)
        assert isinstance(result.filter.filters[1], AndFilter)


class TestFromQueryStringUrlDecoding:
    """Tests for URL decoding of special characters (Requirement 4.8)."""

    def test_url_decode_space(self) -> None:
        """Test URL decoding of space character."""
        result = from_query_string("name=eq.alice%20smith")
        assert isinstance(result, ComparisonFilter)
        assert result.value == "alice smith"

    def test_url_decode_ampersand(self) -> None:
        """Test URL decoding of ampersand."""
        result = from_query_string("company=eq.A%26B%20Corp")
        assert isinstance(result, ComparisonFilter)
        assert result.value == "A&B Corp"

    def test_url_decode_equals(self) -> None:
        """Test URL decoding of equals sign."""
        result = from_query_string("equation=eq.x%3Dy")
        assert isinstance(result, ComparisonFilter)
        assert result.value == "x=y"

    def test_url_decode_parentheses(self) -> None:
        """Test URL decoding of parentheses."""
        result = from_query_string("note=eq.%28important%29")
        assert isinstance(result, ComparisonFilter)
        assert result.value == "(important)"

    def test_url_decode_comma(self) -> None:
        """Test URL decoding of comma."""
        result = from_query_string("list=eq.a%2Cb%2Cc")
        assert isinstance(result, ComparisonFilter)
        assert result.value == "a,b,c"

    def test_url_decode_in_list_values(self) -> None:
        """Test URL decoding in IN operator list values."""
        result = from_query_string("tag=in.(hello%20world,foo%26bar)")
        assert isinstance(result, ComparisonFilter)
        assert result.value == ["hello world", "foo&bar"]

    def test_url_decode_unicode(self) -> None:
        """Test URL decoding of unicode characters."""
        result = from_query_string("name=eq.%E6%97%A5%E6%9C%AC%E8%AA%9E")
        assert isinstance(result, ComparisonFilter)
        assert result.value == "日本語"


class TestFromQueryStringErrors:
    """Tests for error handling in from_query_string."""

    def test_empty_query_raises_value_error(self) -> None:
        """Test that empty query raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("")
        assert "Invalid query string format" in str(exc_info.value)

    def test_missing_equals_raises_value_error(self) -> None:
        """Test that missing equals sign raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("name.eq.alice")
        assert "Invalid query string format" in str(exc_info.value)

    def test_unknown_operator_raises_value_error(self) -> None:
        """Test that unknown operator raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("name=unknown.alice")
        assert "Unknown operator in query string" in str(exc_info.value)

    def test_unbalanced_parentheses_raises_value_error(self) -> None:
        """Test that unbalanced parentheses raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("and=(age.gte.18,status.eq.active")
        assert "Invalid query string format" in str(exc_info.value)

    def test_in_operator_missing_parentheses_raises_value_error(self) -> None:
        """Test that IN operator without parentheses raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("status=in.active,pending")
        assert "Invalid query string format" in str(exc_info.value)

    def test_invalid_nested_format_raises_value_error(self) -> None:
        """Test that invalid nested format raises ValueError."""
        with pytest.raises(ValueError) as exc_info:
            from_query_string("and=(invalid)")
        assert "Invalid query string format" in str(exc_info.value)


class TestFromQueryStringRoundTrip:
    """Tests for round-trip consistency between to_query_string and from_query_string."""

    def test_roundtrip_eq(self) -> None:
        """Test round-trip for eq operator."""
        original = ComparisonFilter.eq("name", "alice")
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original

    def test_roundtrip_in(self) -> None:
        """Test round-trip for in operator."""
        original = ComparisonFilter.in_("status", ["active", "pending"])
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original

    def test_roundtrip_and(self) -> None:
        """Test round-trip for AND filter."""
        original = AndFilter(
            filters=[
                ComparisonFilter.gte("age", 18),
                ComparisonFilter.eq("status", "active"),
            ]
        )
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original

    def test_roundtrip_or(self) -> None:
        """Test round-trip for OR filter."""
        original = OrFilter(
            filters=[
                ComparisonFilter.eq("role", "admin"),
                ComparisonFilter.eq("role", "moderator"),
            ]
        )
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original

    def test_roundtrip_not_comparison(self) -> None:
        """Test round-trip for NOT with ComparisonFilter."""
        original = NotFilter(filter=ComparisonFilter.eq("status", "deleted"))
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original

    def test_roundtrip_not_and(self) -> None:
        """Test round-trip for NOT with AndFilter."""
        original = NotFilter(
            filter=AndFilter(
                filters=[
                    ComparisonFilter.eq("active", False),
                    ComparisonFilter.gt("age", 30),
                ]
            )
        )
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original

    def test_roundtrip_nested(self) -> None:
        """Test round-trip for nested filters."""
        original = OrFilter(
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
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original

    def test_roundtrip_url_encoded(self) -> None:
        """Test round-trip for URL encoded values."""
        original = ComparisonFilter.eq("name", "alice smith")
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original

    def test_roundtrip_special_chars(self) -> None:
        """Test round-trip for special characters."""
        original = ComparisonFilter.eq("company", "A&B Corp")
        query = to_query_string(original)
        parsed = from_query_string(query)
        assert parsed == original
