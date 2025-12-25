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
Unit tests for builtin tools.

This module contains unit tests for each builtin tool in the nexau framework.
Each test should focus on a single tool's functionality in isolation.
"""

# Import existing test modules
# Note: Some modules are placeholders and will be implemented later
try:
    from .test_bash import *
except ImportError:
    pass

try:
    from .test_file_tools import *
except ImportError:
    pass

# Future imports (to be implemented)
# from .test_web_tools import *
# from .test_mcp import *
# from .test_llm_friendly import *
# from .test_other_tools import *

__all__: list[str] = [
    # Add test class names as they are implemented
]
