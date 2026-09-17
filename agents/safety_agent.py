from openai import OpenAI

from agents.config import OPENAI_MODEL


class Obnoxious_Agent:
    def __init__(self, client: OpenAI):
        self.client = client
        self.prompt = (
            "You are a toxicity/harassment classifier.\n"
            "Return ONLY one word: 'Yes' or 'No'.\n"
            "Answer 'Yes' if the message includes ANY of the following:\n"
            "- direct insults or name-calling (e.g., idiot, moron, fool, dumb, stupid, useless, trash, clown, loser)\n"
            "- profanity directed at someone (e.g., 'shut up', 'what the hell are you', etc.)\n"
            "- harassment/abuse, hate, sexual content, or violence/threats\n"
            "Otherwise answer 'No'.\n"
            "Be strict: if there is an insult or abusive phrasing, answer 'Yes' even if the rest is a normal question."
        )

    def extract_action(self, response: str) -> bool:
        return response.strip().lower().startswith("yes")

    def check_query(self, user_text: str) -> bool:
        messages = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": user_text},
        ]
        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.0,
        )
        return self.extract_action(resp.choices[0].message.content)
