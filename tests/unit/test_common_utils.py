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

"""Unit tests for common utility functions."""

import asyncio
import concurrent.futures

import pytest

from nexau.archs.main_sub.utils.common import run_sync


class TestRunSync:
    """Test cases for run_sync function."""

    def test_run_sync_no_running_loop(self):
        """Test run_sync when no event loop is running."""

        async def simple_coro():
            return "result"

        result = run_sync(simple_coro())
        assert result == "result"

    def test_run_sync_with_async_operations(self):
        """Test run_sync with actual async operations."""

        async def async_operation():
            await asyncio.sleep(0.01)
            return 42

        result = run_sync(async_operation())
        assert result == 42

    def test_run_sync_with_exception(self):
        """Test run_sync propagates exceptions from coroutine."""

        async def failing_coro():
            raise ValueError("test error")

        with pytest.raises(ValueError, match="test error"):
            run_sync(failing_coro())

    def test_run_sync_with_timeout(self):
        """Test run_sync with timeout parameter."""

        async def quick_coro():
            return "quick"

        result = run_sync(quick_coro(), timeout=5.0)
        assert result == "quick"

    def test_run_sync_inside_running_loop(self):
        """Test run_sync when called from within a running event loop."""

        async def inner_coro():
            return "inner_result"

        async def outer_coro():
            # This simulates calling run_sync from within an async context
            # run_sync should use thread pool to avoid deadlock
            return run_sync(inner_coro(), timeout=5.0)

        # Run the outer coroutine which calls run_sync internally
        result = asyncio.run(outer_coro())
        assert result == "inner_result"

    def test_run_sync_timeout_exceeded(self):
        """Test run_sync raises TimeoutError when timeout is exceeded in nested loop context."""
        import asyncio

        async def slow_coro():
            await asyncio.sleep(10)
            return "slow"

        async def outer():
            # Inside a running loop, run_sync uses thread pool with timeout
            with pytest.raises(concurrent.futures.TimeoutError):
                run_sync(slow_coro(), timeout=0.01)

        asyncio.run(outer())


class TestImportFromString:
    """Test cases for import_from_string function."""

    def test_import_from_string_success(self):
        """Test successful import from string."""
        from nexau.archs.main_sub.utils.common import import_from_string

        # Import a known function
        result = import_from_string("os.path:join")
        import os.path

        assert result is os.path.join

    def test_import_from_string_missing_separator(self):
        """Test import_from_string raises error when separator is missing."""
        from nexau.archs.main_sub.utils.common import ConfigError, import_from_string

        with pytest.raises(ConfigError, match="Import string must contain"):
            import_from_string("os.path.join")

    def test_import_from_string_module_not_found(self):
        """Test import_from_string raises error when module not found."""
        from nexau.archs.main_sub.utils.common import ConfigError, import_from_string

        with pytest.raises(ConfigError, match="Could not import module"):
            import_from_string("nonexistent.module:function")

    def test_import_from_string_attribute_not_found(self):
        """Test import_from_string raises error when attribute not found."""
        from nexau.archs.main_sub.utils.common import ConfigError, import_from_string

        with pytest.raises(ConfigError, match="Could not import attribute"):
            import_from_string("os.path:nonexistent_function")

    def test_import_from_string_class(self):
        """Test importing a class from string."""
        from nexau.archs.main_sub.utils.common import import_from_string

        result = import_from_string("collections:OrderedDict")
        from collections import OrderedDict

        assert result is OrderedDict


class TestLoadYamlWithVars:
    """Test cases for load_yaml_with_vars function."""

    def test_load_yaml_with_env_var_not_set(self, tmp_path):
        """Test load_yaml_with_vars raises error when env var not set."""
        from nexau.archs.main_sub.utils.common import ConfigError, load_yaml_with_vars

        config_path = tmp_path / "test.yaml"
        with open(config_path, "w") as f:
            f.write("key: ${env.NONEXISTENT_VAR_12345}")

        with pytest.raises(ConfigError, match="Environment variable.*is not set"):
            load_yaml_with_vars(config_path)
