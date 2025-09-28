# app/models/llm/local_model.py
import logging
import time
from llama_cpp import Llama
from app.core.config import settings

logger = logging.getLogger("local_model")

_llm = None


def load_local_model(
    model_path: str = None, n_ctx: int = 4096, n_threads: int = 8, n_gpu_layers: int = 0
):
    model_path = model_path or settings.MODEL_PATH
    logger.info("Loading local LLaMA model from %s", model_path)
    start = time.time()
    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )
    logger.info("Loaded model in %.2f seconds", time.time() - start)
    return llm


def get_llm():
    global _llm
    if _llm is None:
        _llm = load_local_model()
    return _llm


def local_summarize_text(
    text: str,
    max_sentences: int = 2,
    include_examples: bool = False,
    subject: str = "general",
):
    llm = get_llm()
    prompt = (
        f"Summarize the following content into clear, structured Markdown study notes ({max_sentences} sentences):\n\n"
        f"{text}\n\n"
        "Instructions:\n"
        "- Use headings and bullet points.\n"
        "- Use simple, student-friendly language.\n"
        "- Remove repetition and irrelevant details.\n"
        "- Present examples or analogies if needed.\n\n"
        "Output only the structured notes:"
    )

    if include_examples:
        prompt += (
            "Also provide a small example or analogy appropriate for students.\n\n"
        )
    prompt += "Summary:"
    resp = llm(prompt, max_tokens=256, temperature=0.3)
    return resp["choices"][0]["text"].strip()


def merge_with_local_llama(
    summaries, include_examples: bool = False, subject: str = "general"
):
    llm = get_llm()
    batch_text = "\n\n".join(summaries)
    prompt = (
        "Merge these summaries into one structured Markdown study note:\n\n"
        f"{batch_text}\n\n"
        "Instructions:\n"
        "- Use headings (#, ##, ###) and bullet points.\n"
        "- Keep it simple, clear, and student-friendly.\n"
        "- Remove duplicates and redundant info.\n"
        "- Add short examples or analogies if relevant.\n\n"
        "Output only the final merged notes:"
    )

    if include_examples:
        prompt += "Add small illustrative examples or analogies where appropriate.\n\n"
    prompt += "Notes:"
    resp = llm(prompt, max_tokens=512, temperature=0.4)
    return resp["choices"][0]["text"].strip()
