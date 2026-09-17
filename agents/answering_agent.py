from openai import OpenAI

from agents.config import OPENAI_MODEL, REFUSAL_PREFIX


class Answering_Agent:
    def __init__(self, client: OpenAI):
        self.client = client

    def generate_response(self, query: str, docs):
        docs_text = "\n\n".join([d.page_content for d in docs])

        system_prompt = (
            "You answer questions about a Machine Learning document.\n"
            "Use ONLY the provided document excerpts.\n"
            "If the excerpts contain enough information to answer ANY part of the question, answer that part.\n"
            "If a part is not supported by the excerpts, explicitly say you cannot answer that part from the document.\n"
            f"If NONE of the question can be answered from the excerpts, say exactly: "
            f"'{REFUSAL_PREFIX} irrelevant — I do not have enough relevant information in the document to answer that.'\n"
            "Keep the response concise."
            "Do not mention or discuss any non-machine-learning requests. Ignore them silently.\n"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question:\n{query}\n\nDocument excerpts:\n{docs_text}"},
        ]

        resp = self.client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            temperature=0.3,
        )

        return resp.choices[0].message.content.strip()
