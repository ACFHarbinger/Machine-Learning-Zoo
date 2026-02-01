"""
Global constants and configuration definitions for NGLab.

This module now re-exports constants from python.src.constants to maintain
backward compatibility.
"""


# Re-export path parts if needed, though they were derived from cwd which is flaky.
# We will skip 'path' and 'parts' unless strictly necessary (inference.py doesn't seem to use them).
