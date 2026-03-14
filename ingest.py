#!/usr/bin/env python3
"""
Document ingestion — text extraction and LLM proposition extraction.

Two modes:
  ephemeral  — raw text injected into the current conversation's system prompt only
  persistent — LLM extracts atomic propositions stored permanently in memory
"""

import io
import json
from pathlib import Path

import httpx


# ---------------------------------------------------------------------------
# LLM prompt
# ---------------------------------------------------------------------------

INGEST_SYSTEM_PROMPT = """\
You are a knowledge extraction system. Analyse the document provided and extract ALL meaningful information as atomic propositions, organised into logical sections.

Return ONLY a valid JSON object with this exact structure:
{
  "doc_summary": "One sentence describing the entire document",
  "sections": [
    {
      "name": "Section Name",
      "summary": "One sentence describing what this section covers",
      "propositions": [
        {
          "summary": "A single atomic, self-contained factual statement (max 180 chars)",
          "category": "fact|procedure|definition|reference|warning|specification",
          "priority": 3
        }
      ]
    }
  ]
}

Rules:
- Every proposition must be self-contained — a reader with no prior context can fully understand it on its own
- Include the subject explicitly in each proposition (write "The BC250 cluster supports..." not "It supports...")
- Maximum 180 characters per proposition summary
- Priority scale: 5=critical, 4=important, 3=useful, 2=minor (omit 1-2)
- Minimum priority to include: 3
- Group propositions into sections that match the document's natural structure; if the document has no sections, create logical groupings yourself
- Order propositions within each section in logical reading order (prerequisites before dependents)
- Do not invent information not present in the document
"""


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------

def extract_text(filename: str, data: bytes) -> str:
    """
    Extract plain text from file bytes. Returns empty string on failure.
    Tries pymupdf then pypdf for PDFs; python-docx for DOCX; UTF-8 for everything else.
    """
    ext = Path(filename).suffix.lower()

    if ext in (".txt", ".md", ".csv", ".log", ".rst", ".text"):
        return data.decode("utf-8", errors="replace")

    if ext == ".pdf":
        # Prefer pymupdf (faster, better layout), fall back to pypdf
        try:
            import pymupdf  # type: ignore
            doc = pymupdf.open(stream=data, filetype="pdf")
            pages = [page.get_text() for page in doc]
            return "\n\n".join(p for p in pages if p.strip())
        except Exception:
            pass
        try:
            import pypdf  # type: ignore
            reader = pypdf.PdfReader(io.BytesIO(data))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n\n".join(p for p in pages if p.strip())
        except Exception:
            return ""

    if ext == ".docx":
        try:
            import docx  # type: ignore
            doc = docx.Document(io.BytesIO(data))
            return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())
        except Exception:
            return ""

    # Generic fallback — try UTF-8
    return data.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# LLM proposition extraction
# ---------------------------------------------------------------------------

async def ingest_document(
    text: str,
    doc_name: str,
    client: httpx.AsyncClient,
    llm_url: str,
    model: str,
) -> dict:
    """
    Run LLM proposition extraction on document text.

    With a 32K-token context window (~128K chars), most documents fit in one
    pass. We reserve ~8K tokens for the output and system prompt, leaving
    ~24K tokens (~96K chars) for the document body.

    Returns the parsed ingestion result dict:
      { doc_name, doc_summary, sections: [ { name, summary, propositions: [...] } ] }

    Raises httpx.HTTPStatusError or json.JSONDecodeError on failure.
    """
    max_chars = 96_000
    truncated = False
    if len(text) > max_chars:
        text = text[:max_chars]
        truncated = True

    user_content = f"Document name: {doc_name}\n\n---\n\n{text}"
    if truncated:
        user_content += f"\n\n[Document truncated — only the first {max_chars} characters were processed]"

    response = await client.post(
        f"{llm_url}/v1/chat/completions",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": INGEST_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            "max_tokens": 8000,
            "stream": False,
            "response_format": {"type": "json_object"},
        },
        timeout=180.0,
    )
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"].strip()

    # Strip markdown code fences if the model wraps the JSON anyway
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1]
        if raw.endswith("```"):
            raw = raw[:-3].strip()

    result = json.loads(raw)
    result["doc_name"] = doc_name
    result.setdefault("doc_summary", "")
    result.setdefault("sections", [])
    return result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def count_propositions(ingestion_result: dict) -> int:
    return sum(
        len(s.get("propositions", []))
        for s in ingestion_result.get("sections", [])
    )
