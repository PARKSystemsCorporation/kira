from __future__ import annotations

from typing import Any, Dict, List, Optional


def _extract_keywords(text: str) -> List[str]:
    stopwords = {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "can", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why", "how",
        "all", "each", "few", "more", "most", "other", "some", "such",
        "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "just", "and", "but", "if", "or", "because", "as", "until", "while",
        "i", "me", "my", "myself", "we", "our", "you", "your", "he", "him",
        "she", "her", "it", "its", "they", "them", "what", "which", "who",
        "this", "that", "these", "those", "am", "know", "tell", "about",
        "kira", "hey", "please", "thanks", "thank",
    }

    words = (
        text.lower()
        .replace("'", " ")
    )
    words = [w for w in "".join([c if c.isalnum() or c.isspace() else " " for c in words]).split()]
    return [w for w in words if len(w) > 2 and w not in stopwords]


class Router:
    def __init__(self, config: Dict[str, Any]) -> None:
        self._config = config

    def build_prompt(
        self,
        message: str,
        memory_context: Dict[str, List[Dict[str, Any]]],
        system: Optional[str] = None,
    ) -> str:
        context = self._build_context_string(message, memory_context)

        if system:
            return f"{system}\n\n{context}".strip()

        from .prompts import _get_prompt

        template = _get_prompt("default")
        return template.format(context=context)

    def _build_context_string(
        self, user_message: str, memory_context: Dict[str, List[Dict[str, Any]]]
    ) -> str:
        keywords = _extract_keywords(user_message)

        short_mem = memory_context.get("short", [])
        medium_mem = memory_context.get("medium", [])
        long_mem = memory_context.get("long", [])
        phrases = memory_context.get("phrases", [])

        context = ""
        has_chat_info = bool(short_mem or medium_mem or long_mem or phrases)

        if has_chat_info:
            context += "## Chat Memory\n"
            for corr in (long_mem[:5] + medium_mem[:5] + short_mem[:5]):
                strength = "★★★" if corr.get("tier") == "long" else "★★" if corr.get("tier") == "medium" else "★"
                w1 = corr.get("word1", "")
                w2 = corr.get("word2", "")
                count = corr.get("reinforcement_count", 1)
                context += f"[{strength}] \"{w1}\" + \"{w2}\" (mentioned {count}x)\n"
            if phrases:
                context += "\n## Phrases\n"
                for p in phrases[:5]:
                    context += f"- {p.get('phrase_key', '')}\n"

        if not has_chat_info:
            if keywords:
                context += "## No Relevant Information Found\n"
                context += f"Searched for: {', '.join(keywords)}\n"
            else:
                context += "## No Relevant Information Found\n"

        return context.strip()