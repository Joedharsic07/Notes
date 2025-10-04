import os
import logging
import time
import tiktoken
from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger("gemini_model")

# -------------------------------------
# Gemini Setup
# -------------------------------------
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = settings.GOOGLE_APPLICATION_CREDENTIALS

CLIENT = genai.Client(
    vertexai=True,
    project=settings.PROJECT_ID,
    location=settings.LOCATION
)

MODEL_NAME = "gemini-2.5-flash"

# -------------------------------------
# Prompt Builders
# -------------------------------------
def build_youtube_prompt(text: str, max_sentences: int = 5, include_examples: bool = False) -> str:
    """
    Prompt for summarizing YouTube transcript or long text into Markdown notes.
    """
    prompt = f"""
You are an expert educational content summarizer. Summarize the given text into clear, structured, student-friendly Markdown notes.

Input Text:
{text}

Instructions:
1. Summarize each main idea in about {max_sentences} sentences.
2. Use Markdown:
   - # for main topics
   - ## for subtopics
   - - for bullet points
3. Keep explanations clear and easy for students to understand.
4. Remove filler words, repetitions, and timestamps.
5. Retain key terms, formulas, and examples.
6. Include examples or analogies only if include_examples=True.
7. Ensure valid, clean Markdown output (no unnecessary symbols or escapes).
"""
    if include_examples:
        prompt += "\nInclude small relatable examples where appropriate.\n"
    return prompt.strip()


def build_notes_prompt(text: str, include_examples: bool = False) -> str:
    """
    Prompt for uploaded notes rewriting — preserve structure, simplify explanations,
    and ensure Markdown readability.
    """
    prompt = f"""
You are an expert study assistant. Rewrite the following educational notes into simple, easy-to-understand Markdown for students.

Input Notes:
{text}

Instructions:
1. Start rewriting only from actual lesson content (Units/Chapters).
2. Ignore preface, TOC, page numbers, and repeated college headers.
3. Keep structure and hierarchy:
   - # for Unit/Chapter titles
   - ## for subtopics
   - - for bullet points
   - Use ``` for code/formulas if needed
4. Simplify complex sentences.
5. Expand unclear statements briefly for clarity.
6. Preserve **all examples, formulas, and definitions**.
   - If examples are unclear, rewrite clearly.
   - If none and include_examples=True, add one illustrative example.
7. Ensure valid Markdown (no escape characters or broken syntax).
8. Do not skip content — just make it clearer and student-friendly.
"""
    if include_examples:
        prompt += "\nInclude small illustrative examples where relevant.\n"
    return prompt.strip()

# -------------------------------------
# Markdown Cleaner
# -------------------------------------
def clean_markdown(text: str) -> str:
    """
    Cleans the generated text so it renders properly in Markdown viewers.
    """
    if not text:
        return ""

    text = (
        text.replace("\\n", "\n")
        .replace("\\`", "`")
        .replace("** **", "**")
        .replace("  *", "-")
        .replace("  -", "-")
        .replace("\r", "")
    )

    lines = [line.strip() for line in text.splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    return cleaned.strip()

# -------------------------------------
# Token Tools
# -------------------------------------
def count_tokens(text: str) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

def split_text_smart(text: str, max_tokens: int = 90_000):
    """
    Split text intelligently based on token limits, avoiding mid-sentence breaks.
    """
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(text)

    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = encoding.decode(chunk_tokens)

        last_sentence_end = max(chunk_text.rfind("."), chunk_text.rfind("\n"))
        if last_sentence_end > 0 and end != len(tokens):
            chunk_text = chunk_text[:last_sentence_end + 1]

        chunks.append(chunk_text)
        start += len(encoding.encode(chunk_text))

    return chunks

# -------------------------------------
# Gemini API Wrapper
# -------------------------------------
def generate_content(prompt: str, max_output_tokens: int = 60000, temperature: float = 0.3):
    """
    Calls Gemini API and returns cleaned text with token usage stats.
    """
    try:
        start_time = time.time()
        response = CLIENT.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )
        duration = time.time() - start_time

        # Extract text safely
        raw_text = ""
        if hasattr(response, "text") and response.text:
            raw_text = response.text
        elif hasattr(response, "candidates") and response.candidates:
            raw_text = response.candidates[0].content.parts[0].text

        cleaned_text = clean_markdown(raw_text)

        # Token usage info
        usage = getattr(response, "usage_metadata", None)
        token_usage = {
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "total_tokens": getattr(usage, "total_tokens", None),
            "generation_time": round(duration, 2),
        }

        return cleaned_text, token_usage

    except Exception as e:
        logger.error("Gemini LLM call failed: %s", e)
        return f"[Error: {e}]", {"error": str(e)}

# -------------------------------------
# YouTube Summarization
# -------------------------------------
def generate_youtube_summary(text: str, max_sentences: int = 5, include_examples: bool = False, delay: float = 0.3):
    """
    Summarize long YouTube transcripts or articles into Markdown notes.
    Splits content intelligently into token-safe chunks.
    """
    chunks = split_text_smart(text)
    summaries = []
    total_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}

    for i, chunk in enumerate(chunks, start=1):
        logger.info("Summarizing chunk %d/%d", i, len(chunks))
        prompt = build_youtube_prompt(chunk, max_sentences=max_sentences, include_examples=include_examples)
        summary, usage = generate_content(prompt)
        summaries.append(summary)

        # Aggregate token usage
        for k in total_usage.keys():
            if k in usage and isinstance(usage[k], (int, float)):
                total_usage[k] += usage[k]

        time.sleep(delay)

    merged_summary = "\n\n".join(summaries)
    return clean_markdown(merged_summary), total_usage

# -------------------------------------
# Notes Rewriting (Full)
# -------------------------------------
def rewrite_notes_full(text: str, include_examples: bool = False):
    """
    Rewrite a full uploaded notes document, preserving structure and returning token usage.
    """
    prompt = build_notes_prompt(text, include_examples)
    rewritten_text, token_usage = generate_content(prompt, max_output_tokens=65000)
    return rewritten_text, token_usage
