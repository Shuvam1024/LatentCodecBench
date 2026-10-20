"""Commit recorded when this distribution was built.

An editable checkout leaves ``GIT_SHA`` empty. A wheel built from a git
checkout replaces it with ``git rev-parse HEAD`` and restores this file
after the build.
"""

GIT_SHA = ""
