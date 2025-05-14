#!/usr/bin/env python3
"""
generate_whitecontext_outreach.py – Generates personalized LinkedIn outreach
messages written in the voice of Maksym Huczynski (CTO, WhiteContext) that
invite Y Combinator founders to collaborate on a public showcase combining
WhiteContext technology with their own solution.

Input files
-----------
1. **ai_founders_with_llms.csv** – Must include the columns:
   ``Founder Name``, ``Company Name``, ``Company Website URL``,
   ``llms_txt_overview`` (startup‑specific website synthesis).
2. **whitecontext_llms.txt** – A concise overview of WhiteContext’s
   technology (can be produced by the same crawler that generated the other
   *llms.txt* files).

Output file
-----------
* **yc_founders_whitecontext_outreach.csv** – Extends the input rows with
  ``connection_request``, ``follow_up_message``, and ``constraint_issues``.

Requirements
------------
* pandas
* python‑dotenv  (for environment variables)
* google‑generativeai (Gemini API client)

Environment variables
---------------------
* ``GEMINI_API_KEY`` – your Google Gemini key

Notes
-----
* LinkedIn invitation **must be ≤ 160 characters** (LinkedIn limit).
* Follow‑up DM **must be ≤ 40 words**.
* Both messages must reference at least one concrete element from the
  startup’s *llms_txt_overview* **and** one concrete element from
  *whitecontext_llms.txt*.
* The script retries transient Gemini errors with jittered exponential back‑off.
"""

from __future__ import annotations

import os
import random
import time
from pathlib import Path
from typing import Tuple, List

import pandas as pd
from dotenv import load_dotenv
from google import genai
import logging

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ROOT_DIR = Path(__file__).resolve().parent
INPUT_CSV = ROOT_DIR / "ai_founders_with_llms.csv"
WHITE_CONTEXT_TXT = ROOT_DIR / "whitecontext_llms.txt"
OUTPUT_CSV = ROOT_DIR / "yc_founders_whitecontext_outreach.csv"

SENDER_NAME = "Maksym Huczynski"
SENDER_TITLE = "CTO at WhiteContext"
SENDER_COMPANY = "WhiteContext"

MAX_CONNECTION_LEN = 160  # characters
MAX_FOLLOWUP_WORDS = 40   # words
MAX_CONTEXT_LEN = 3500    # safety margin for prompt tokens

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s – %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Environment & Gemini client
# ---------------------------------------------------------------------------

load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY not set in environment variables")

client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def retry_on_gemini_errors(max_retries: int = 50, base_delay: int = 1, max_delay: int = 60):
    """Retry decorator for transient Gemini errors."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            retries = 0
            while True:
                try:
                    return func(*args, **kwargs)
                except Exception as exc:  # broad but safe – logged below
                    retries += 1
                    if retries > max_retries:
                        logger.error("Gemini API call failed after %s retries: %s", retries - 1, exc)
                        raise
                    delay = min(base_delay * 2 ** (retries - 1), max_delay)
                    delay *= random.uniform(0.5, 1.5)  # add jitter
                    logger.warning("Gemini API retry %s/%s after %.1fs – %s", retries, max_retries, delay, exc)
                    time.sleep(delay)

        return wrapper

    return decorator


@retry_on_gemini_errors()
def generate_messages(
    founder_name: str,
    company_name: str,
    company_website: str,
    startup_context: str,
    whitecontext_overview: str,
) -> str:
    """Return raw Gemini response with both outreach messages."""

    # Truncate contexts if extremely long (Gemini Flash ~8k tokens limit)
    startup_context = startup_context[:MAX_CONTEXT_LEN]
    whitecontext_overview = whitecontext_overview[:MAX_CONTEXT_LEN]

    prompt = f"""
You are **{SENDER_NAME}, {SENDER_TITLE}**.

**WhiteContext overview:**
{whitecontext_overview}

**Startup overview (from {company_website}):**
{startup_context}

Craft two *LinkedIn* messages to **{founder_name}**, founder of **{company_name}**:

1. **Connection Request** – *≤ {MAX_CONNECTION_LEN} characters (including spaces and punctuation).*
   • Reference a specific strength or feature from {company_name} **and** a specific capability of {SENDER_COMPANY}.
   • End with a friendly invitation to connect.

2. **Follow‑Up Message** – *≤ {MAX_FOLLOWUP_WORDS} words.* Assume they accepted. Invite them to co‑create a **public showcase** demonstrating how their product integrates with WhiteContext technology. Highlight mutual benefit.

Do **not** use placeholders like "[Your name]" – write the actual content. Keep tone professional yet enthusiastic.

Return messages exactly in this markdown template (no extra lines):

CONNECTION REQUEST:
<connection‑request‑text>

FOLLOW‑UP MESSAGE:
<follow‑up‑text>
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash-preview-04-17",
        contents=prompt,
    )

    # Support both .text (google‑generativeai ≥ 0.4.0) and legacy .parts
    if hasattr(response, "text"):
        return response.text
    if hasattr(response, "parts"):
        return "".join(part.text for part in response.parts)
    return str(response)


def parse_response(raw: str) -> Tuple[str, str]:
    """Extract the two message blocks from Gemini response."""
    connection, follow_up = "", ""
    section = None
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.upper().startswith("CONNECTION REQUEST"):
            section = "conn"
            continue
        if line.upper().startswith("FOLLOW-UP MESSAGE") or line.upper().startswith("FOLLOW‑UP MESSAGE"):
            section = "follow"
            continue

        if section == "conn":
            connection += (" " if connection else "") + line
        elif section == "follow":
            follow_up += (" " if follow_up else "") + line

    return connection.strip(), follow_up.strip()


def verify(connection: str, follow_up: str) -> List[str]:
    """Return list of constraint violation strings, empty if none."""
    issues: List[str] = []
    if len(connection) > MAX_CONNECTION_LEN:
        issues.append(f"Invitation length {len(connection)} > {MAX_CONNECTION_LEN} chars")
    if len(follow_up.split()) > MAX_FOLLOWUP_WORDS:
        issues.append(f"Follow‑up length {len(follow_up.split())} words > {MAX_FOLLOWUP_WORDS}")
    return issues

# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main() -> None:
    if not INPUT_CSV.exists():
        logger.error("Input CSV %s not found", INPUT_CSV)
        return

    if not WHITE_CONTEXT_TXT.exists():
        logger.error("WhiteContext overview file %s not found", WHITE_CONTEXT_TXT)
        return

    whitecontext_overview = WHITE_CONTEXT_TXT.read_text(encoding="utf‑8").strip()
    if not whitecontext_overview:
        logger.error("WhiteContext overview file is empty – aborting")
        return

    df = pd.read_csv(INPUT_CSV)

    for col in (
        "Founder Name",
        "Company Name",
        "Company Website URL",
        "llms_txt_overview",
    ):
        if col not in df.columns:
            raise ValueError(f"CSV missing required column: {col}")

    df["connection_request"] = ""
    df["follow_up_message"] = ""
    df["constraint_issues"] = ""

    total = len(df)
    logger.info("Processing %s founders", total)

    for idx, row in df.iterrows():
        founder_name = str(row["Founder Name"]).strip()
        company_name = str(row["Company Name"]).strip()
        company_site = str(row["Company Website URL"]).strip()
        startup_ctx = str(row.get("llms_txt_overview", "")).replace("\\n", "\n").strip()

        if not (founder_name and company_name and company_site):
            issue = "Missing required data"
            df.at[idx, "constraint_issues"] = issue
            logger.warning("Row %s skipped – %s", idx, issue)
            continue

        try:
            raw_response = generate_messages(
                founder_name,
                company_name,
                company_site,
                startup_ctx,
                whitecontext_overview,
            )
            conn, follow = parse_response(raw_response)
            issues = verify(conn, follow)

            df.at[idx, "connection_request"] = conn
            df.at[idx, "follow_up_message"] = follow
            df.at[idx, "constraint_issues"] = "; ".join(issues)

            if issues:
                logger.warning("Row %s – constraint issues: %s", idx, issues)

        except Exception as exc:
            logger.exception("Error processing row %s (%s): %s", idx, founder_name, exc)
            df.at[idx, "constraint_issues"] = f"Error: {exc}"[:120]

        # Persist every 5 rows or on last row
        if (idx + 1) % 5 == 0 or idx == total - 1:
            df.to_csv(OUTPUT_CSV, index=False)
            logger.info("Progress saved – %s/%s rows", idx + 1, total)

    logger.info("🎉 Outreach generation complete – results in %s", OUTPUT_CSV)

    ok = (df["constraint_issues"] == "").sum()
    bad = total - ok
    logger.info("Summary: %s success, %s with issues", ok, bad)


if __name__ == "__main__":
    main()
