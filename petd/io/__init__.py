"""
Hardware/service adapters. The existing clients (vacuum-api, face-api,
playerc-client) live in hyphenated folders next to this package and are
imported as-is by putting those folders on sys.path.
"""

import sys

from ..config import PROJECT_ROOT

for _folder in ("vacuum-api", "face-api", "playerc-client"):
    _path = str(PROJECT_ROOT / _folder)
    if _path not in sys.path:
        sys.path.insert(0, _path)
