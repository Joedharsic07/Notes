import google.generativeai as genai
import logging
from app.core.config import settings

class GeminiModel:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-flash"):
        if not api_key:
            raise ValueError("Gemini API key is required")
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)

    def summarize(self, text: str, max_sentences: int = 5) -> str:
        prompt = (
            f"Summarize the following content in a structured and educational way "
            f"into {max_sentences} sentences:\n\n"
            f"{text}\n\n"
            "Instructions:\n"
            "- Use bullet points and appropriate headings.\n"
            "- Remove filler words, repetition, and irrelevant details.\n"
            "- Keep the language simple, direct, and suitable for students.\n"
            "- Make it clear, structured, and student-friendly.\n\n"
            "Summary:"
        )
        response = self.model.generate_content(prompt)
        return response.text.strip()

    def merge(self, summaries: list) -> str:
        prompt = (
            "You are an assistant that merges multiple transcript summaries into one structured Markdown study note.\n"
            "Instructions:\n"
            "- Use headings and subheadings (#, ##, ###).\n"
            "- Use simple words and full sentences.\n"
            "- Present it like you're explaining to a middle school student.\n"
            "- Avoid technical jargon.\n"
            "- Keep the tone friendly and educational.\n"
            "- Use short paragraphs (2-3 sentences each).\n"
            "- Remove duplicates and redundant information.\n"
            "- Make it student-friendly notes .\n\n"
            f"{' '.join(summaries)}\n\nNotes:"
        )
        response = self.model.generate_content(prompt)
        return response.text.strip()
