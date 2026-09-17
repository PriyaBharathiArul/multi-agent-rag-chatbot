from openai import OpenAI

from agents.config import OPENAI_MODEL


class Context_Rewriter_Agent:
    def __init__(self, client: OpenAI):
        self.client = client

    def rephrase(self, history_messages: list[dict], latest_query: str) -> str:
        system_prompt = (
            "You are a query rewriting assistant.\n"
            "Rewrite the user's latest question into a clear standalone question.\n"
            "Do NOT answer it.\n"
            "Return ONLY the rewritten question."
        )

        trimmed = history_messages[-6:] if len(history_messages) > 6 else history_messages
        history_text = "\n".join([f"{m['role']}: {m['content']}" for m in trimmed])

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Conversation history:\n{history_text}\n\nLatest question:\n{latest_query}"},
        ]

        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.2,
        )
        return resp.choices[0].message.content.strip()
