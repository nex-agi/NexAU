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

"""Public API for the config package with lazy imports to avoid cycles."""

from importlib import import_module

__all__ = ["load_agent_config", "ConfigError"]


def __getattr__(name):
    if name in __all__:
        module = import_module("nexau.archs.config.config_loader")
        return getattr(module, name)
    raise AttributeError(f"module 'nexau.archs.config' has no attribute '{name}'")
