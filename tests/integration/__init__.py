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

"""
Integration tests for nexau framework components.

This module contains integration tests that test how different components
work together. These tests may use mocked external dependencies but test
the actual interactions between internal components.
"""

# Import existing test modules
try:
    from .test_agent_execution import *
except ImportError:
    pass

try:
    from .test_tool_integration import *
except ImportError:
    pass

try:
    from .test_config_integration import *
except ImportError:
    pass

__all__: list[str] = [
    # Add test class names as they are implemented
]
