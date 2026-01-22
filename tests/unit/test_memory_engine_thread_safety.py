"""Test thread safety of InMemoryDatabaseEngine.

This module tests that InMemoryDatabaseEngine is safe for concurrent access
from multiple asyncio tasks.
"""

import asyncio

import pytest
from sqlmodel import Field, SQLModel

from nexau.archs.session.orm import ComparisonFilter, InMemoryDatabaseEngine


class Counter(SQLModel, table=True):
    """Simple counter model for testing."""

    __tablename__ = "counters"  # type: ignore[assignment]

    id: str = Field(primary_key=True)
    value: int = 0


@pytest.mark.anyio
async def test_concurrent_create_no_race_condition():
    """Test that concurrent creates don't cause race conditions."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create 100 counters concurrently
    async def create_counter(i: int):
        counter = Counter(id=f"counter_{i}", value=i)
        await engine.create(counter)

    tasks = [create_counter(i) for i in range(100)]
    await asyncio.gather(*tasks)

    # Verify all counters were created
    counters = await engine.find_many(Counter)
    assert len(counters) == 100
    assert sum(c.value for c in counters) == sum(range(100))


@pytest.mark.anyio
async def test_concurrent_update_no_race_condition():
    """Test that concurrent updates don't cause race conditions."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create initial counter
    counter = Counter(id="shared", value=0)
    await engine.create(counter)

    # Update counter concurrently 100 times
    async def increment_counter():
        # Read current value
        current = await engine.find_first(Counter, filters=ComparisonFilter.eq("id", "shared"))
        assert current is not None
        # Increment and update
        current.value += 1
        await engine.update(current)

    tasks = [increment_counter() for _ in range(100)]
    await asyncio.gather(*tasks)

    # Verify final value
    final = await engine.find_first(Counter, filters=ComparisonFilter.eq("id", "shared"))
    assert final is not None
    assert final.value == 100


@pytest.mark.anyio
async def test_concurrent_delete_no_race_condition():
    """Test that concurrent deletes don't cause race conditions."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create 100 counters
    for i in range(100):
        await engine.create(Counter(id=f"counter_{i}", value=i))

    # Delete counters concurrently
    async def delete_counter(i: int):
        await engine.delete(Counter, filters=ComparisonFilter.eq("id", f"counter_{i}"))

    tasks = [delete_counter(i) for i in range(100)]
    await asyncio.gather(*tasks)

    # Verify all counters were deleted
    counters = await engine.find_many(Counter)
    assert len(counters) == 0


@pytest.mark.anyio
async def test_concurrent_mixed_operations():
    """Test concurrent mix of create, read, update, delete operations."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create initial counters
    for i in range(10):
        await engine.create(Counter(id=f"counter_{i}", value=i))

    async def create_op(i: int):
        await engine.create(Counter(id=f"new_{i}", value=i))

    async def read_op(i: int):
        await engine.find_first(Counter, filters=ComparisonFilter.eq("id", f"counter_{i % 10}"))

    async def update_op(i: int):
        counter = await engine.find_first(Counter, filters=ComparisonFilter.eq("id", f"counter_{i % 10}"))
        if counter:
            counter.value += 1
            await engine.update(counter)

    async def delete_op(i: int):
        if i < 5:  # Only delete first 5
            await engine.delete(Counter, filters=ComparisonFilter.eq("id", f"counter_{i}"))

    # Mix of operations
    tasks = []
    for i in range(20):
        tasks.append(create_op(i))
        tasks.append(read_op(i))
        tasks.append(update_op(i))
    for i in range(5):
        tasks.append(delete_op(i))

    await asyncio.gather(*tasks)

    # Verify state
    counters = await engine.find_many(Counter)
    # Should have: 5 original (not deleted) + 20 new = 25
    assert len(counters) == 25


@pytest.mark.anyio
async def test_concurrent_find_many_consistent():
    """Test that concurrent find_many operations return consistent results."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create counters
    for i in range(50):
        await engine.create(Counter(id=f"counter_{i}", value=i))

    # Read concurrently
    async def read_all():
        return await engine.find_many(Counter)

    tasks = [read_all() for _ in range(20)]
    results = await asyncio.gather(*tasks)

    # All reads should return the same count
    for result in results:
        assert len(result) == 50


@pytest.mark.anyio
async def test_concurrent_count_consistent():
    """Test that concurrent count operations return consistent results."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create counters
    for i in range(30):
        await engine.create(Counter(id=f"counter_{i}", value=i))

    # Count concurrently
    async def count_all():
        return await engine.count(Counter)

    tasks = [count_all() for _ in range(20)]
    results = await asyncio.gather(*tasks)

    # All counts should be the same
    for result in results:
        assert result == 30


@pytest.mark.anyio
async def test_no_duplicate_key_error_under_concurrency():
    """Test that duplicate key errors are properly raised under concurrency."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Try to create the same counter concurrently
    async def create_same_counter():
        try:
            await engine.create(Counter(id="shared", value=1))
            return True
        except ValueError as e:
            if "Duplicate primary key" in str(e):
                return False
            raise

    tasks = [create_same_counter() for _ in range(10)]
    results = await asyncio.gather(*tasks)

    # Exactly one should succeed
    assert sum(results) == 1

    # Verify only one counter exists
    counters = await engine.find_many(Counter)
    assert len(counters) == 1


def test_get_shared_instance_returns_singleton():
    """Test that get_shared_instance returns the same instance."""
    # Clear global state for test isolation
    import nexau.archs.session.orm.memory_engine as mem_module

    mem_module._shared_instance = None

    instance1 = InMemoryDatabaseEngine.get_shared_instance()
    instance2 = InMemoryDatabaseEngine.get_shared_instance()

    assert instance1 is instance2


def test_get_shared_instance_different_from_new_instance():
    """Test that get_shared_instance returns different instance from constructor."""
    # Clear global state for test isolation
    import nexau.archs.session.orm.memory_engine as mem_module

    mem_module._shared_instance = None

    shared = InMemoryDatabaseEngine.get_shared_instance()
    new_instance = InMemoryDatabaseEngine()

    assert shared is not new_instance


@pytest.mark.anyio
async def test_find_many_with_offset():
    """Test find_many with offset parameter."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create 10 counters
    for i in range(10):
        await engine.create(Counter(id=f"counter_{i:02d}", value=i))

    # Find with offset
    results = await engine.find_many(Counter, offset=3, order_by="id")

    # Should skip first 3
    assert len(results) == 7
    assert results[0].id == "counter_03"


@pytest.mark.anyio
async def test_find_many_with_offset_and_limit():
    """Test find_many with both offset and limit."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create 10 counters
    for i in range(10):
        await engine.create(Counter(id=f"counter_{i:02d}", value=i))

    # Find with offset and limit
    results = await engine.find_many(Counter, offset=2, limit=3, order_by="id")

    # Should skip first 2 and take next 3
    assert len(results) == 3
    assert results[0].id == "counter_02"
    assert results[2].id == "counter_04"


@pytest.mark.anyio
async def test_find_many_with_multiple_order_by():
    """Test find_many with multiple order_by fields."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create counters with same value but different ids
    await engine.create(Counter(id="b", value=1))
    await engine.create(Counter(id="a", value=1))
    await engine.create(Counter(id="c", value=2))

    # Order by value then id
    results = await engine.find_many(Counter, order_by=("value", "id"))

    assert len(results) == 3
    assert results[0].id == "a"  # value=1, id='a'
    assert results[1].id == "b"  # value=1, id='b'
    assert results[2].id == "c"  # value=2


@pytest.mark.anyio
async def test_find_many_with_descending_order():
    """Test find_many with descending order."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    for i in range(5):
        await engine.create(Counter(id=f"counter_{i}", value=i))

    # Order descending by id (string field)
    results = await engine.find_many(Counter, order_by="-id")

    assert len(results) == 5
    assert results[0].id == "counter_4"
    assert results[4].id == "counter_0"


@pytest.mark.anyio
async def test_create_many():
    """Test create_many method."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    counters = [Counter(id=f"counter_{i}", value=i) for i in range(5)]
    created = await engine.create_many(counters)

    assert len(created) == 5

    # Verify all were saved
    loaded = await engine.find_many(Counter)
    assert len(loaded) == 5


@pytest.mark.anyio
async def test_create_many_duplicate_raises():
    """Test create_many raises on duplicate primary key."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create one first
    await engine.create(Counter(id="counter_0", value=0))

    # Try to create_many with duplicate
    counters = [
        Counter(id="counter_1", value=1),
        Counter(id="counter_0", value=0),  # Duplicate
    ]

    with pytest.raises(ValueError, match="Duplicate primary key"):
        await engine.create_many(counters)


@pytest.mark.anyio
async def test_get_table_auto_creates():
    """Test _get_table auto-creates table if not exists."""
    engine = InMemoryDatabaseEngine()
    # Don't call setup_models

    # Access table through find_many which calls _get_table
    results = await engine.find_many(Counter)

    assert results == []
    # Table should now exist
    assert "counters" in engine._storage


@pytest.mark.anyio
async def test_find_first_no_match():
    """Test find_first returns None when no records match."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create some counters
    await engine.create(Counter(id="counter_1", value=1))

    # Find with filter that matches nothing
    result = await engine.find_first(Counter, filters=ComparisonFilter.eq("id", "nonexistent"))

    assert result is None


@pytest.mark.anyio
async def test_find_many_with_filter():
    """Test find_many with filter parameter."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create some counters
    await engine.create(Counter(id="counter_1", value=1))
    await engine.create(Counter(id="counter_2", value=2))
    await engine.create(Counter(id="counter_3", value=1))

    # Find with filter
    results = await engine.find_many(Counter, filters=ComparisonFilter.eq("value", 1))

    assert len(results) == 2
    assert all(c.value == 1 for c in results)


@pytest.mark.anyio
async def test_count_with_filter():
    """Test count with filter parameter."""
    engine = InMemoryDatabaseEngine()
    await engine.setup_models([Counter])

    # Create some counters
    await engine.create(Counter(id="counter_1", value=1))
    await engine.create(Counter(id="counter_2", value=2))
    await engine.create(Counter(id="counter_3", value=1))

    # Count with filter
    count = await engine.count(Counter, filters=ComparisonFilter.eq("value", 1))

    assert count == 2
