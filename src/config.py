"""I read configuration secrets like the ENTSO-E API key from a local .env file.

I keep secrets out of the source code entirely. The .env file lives only on
my machine, is listed in .gitignore, and is therefore never uploaded to GitHub.
Anyone who clones my repo creates their own .env from .env.example.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# === CONFIGURATION ===

# I load the .env file from the project root, one level above src/.
# load_dotenv() reads each KEY=value line and puts it into os.environ,
# exactly as if it were a real environment variable of the operating system.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


# === ACCESS FUNCTIONS ===

def get_entsoe_api_key() -> str:
    """I return the ENTSO-E API key and raise a clear error if it is missing.

    I raise instead of returning None because a missing key should stop the
    pipeline immediately with a helpful message, not fail later with a
    confusing HTTP 401 error deep inside a fetch loop.
    """
    key = os.environ.get("ENTSOE_API_KEY")
    if not key:
        raise RuntimeError(
            "ENTSOE_API_KEY not found. Create a file named '.env' in the "
            "project root and add one line: ENTSOE_API_KEY=your_key_here"
        )
    return key
