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

import pytest


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
