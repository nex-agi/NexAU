# Copyright (c) Nex-AGI. All rights reserved.

from .aggregated_search import web_search
from .google_web_search import google_web_search
from .web_fetch import web_fetch

__all__ = ["google_web_search", "web_fetch", "web_search"]
