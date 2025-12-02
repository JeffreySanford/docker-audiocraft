import os
import sys

# Ensure the workspace/ directory is on sys.path so tests can import run_large_offload3
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
