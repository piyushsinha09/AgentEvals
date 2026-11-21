import os

# Keep DeepEval quiet and offline-friendly during tests.
os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")
os.environ.setdefault("DEEPEVAL_NO_INSPECT_PROMPT", "1")
