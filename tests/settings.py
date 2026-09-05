# tests/settings.py
import os

TEST_MODE = os.getenv("TEST_MODE", "unit")
SKIP_DB_TESTS = TEST_MODE != "integration"
