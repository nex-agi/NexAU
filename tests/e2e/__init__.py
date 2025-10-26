"""
End-to-end tests for northau framework.

This module contains end-to-end tests that test complete workflows
from start to finish. These tests may use real external services
or comprehensive mocks to test full user scenarios.
"""

# Import existing test modules
try:
    from .test_full_workflows import *
except ImportError:
    pass

__all__ = [
    # Add test class names as they are implemented
]
