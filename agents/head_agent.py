import re

from openai import OpenAI
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

from agents.config import (
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    REFUSAL_PREFIX,
)
from agents.safety_agent import Obnoxious_Agent
from agents.rewriter_agent import Context_Rewriter_Agent
from agents.query_agent import Query_Agent
from agents.answering_agent import Answering_Agent


class Head_Agent:
    """Controller that orchestrates the multi-agent RAG pipeline."""

    def __init__(self, openai_key: str, pinecone_key: str):
        if not openai_key:
            raise ValueError("OPENAI_API_KEY is missing.")
        if not pinecone_key:
            raise ValueError("PINECONE_API_KEY is missing.")

        self.client = OpenAI(api_key=openai_key)

        pc = Pinecone(api_key=pinecone_key)
        self.index = pc.Index(PINECONE_INDEX_NAME)

        self.embeddings = OpenAIEmbeddings(api_key=openai_key)

        self.obnoxious_agent = Obnoxious_Agent(self.client)
        self.rewriter_agent = Context_Rewriter_Agent(self.client)
        self.query_agent = Query_Agent(self.index, self.embeddings, PINECONE_NAMESPACE)
        self.answering_agent = Answering_Agent(self.client)

    def handle_query(self, conversation, prompt):
        debug = {}

        # 1) Safety gate
        if self.obnoxious_agent.check_query(prompt):
            debug["agent_path"] = "obnoxious"
            return (
                f"{REFUSAL_PREFIX} obnoxious — Please rephrase politely and keep it about machine learning.",
                debug,
            )

        # 2) Small-talk bypass
        p = prompt.strip().lower()
        if p in {
            "can you help me with a quick question?",
            "what can you do?",
        }:
            debug["agent_path"] = "small_talk"
            return ("Yes — ask me anything about the machine learning document.", debug)

        p_words = set(re.findall(r"[a-z']+", p))

        small_talk_words = {"hello", "hi", "hey", "thanks", "thank", "bye"}
        small_talk_prefixes = ("good morning", "good afternoon", "good evening")

        if (
            p in {"hello", "hi", "hey", "thanks", "thank you", "bye"}
            or any(p.startswith(pref) for pref in small_talk_prefixes)
            or (len(p_words & small_talk_words) > 0 and len(p_words) <= 4)
        ):
            debug["agent_path"] = "small_talk"
            return (
                "Hello! Ask me anything about the machine learning document and I'll help.",
                debug,
            )

        # 3) Hybrid prompt handling: keep ONLY the ML part
        ml_keywords = [
            "machine learning", "logistic", "sigmoid", "gradient", "overfitting",
            "regularization", "cross-validation", "cross validation",
            "confusion matrix", "precision", "recall", "bias", "variance",
            "decision boundary", "svm", "support vector", "neural network",
            "k-means", "training loss", "validation loss",
            "generative", "discriminative", "supervised", "unsupervised",
            "classification", "regression", "clustering", "embedding",
            "backpropagation", "activation function", "loss function",
            "feature", "hyperparameter", "epoch", "batch size",
            "l1", "l2",
        ]

        lower = prompt.lower()
        separators = [
            " and also ", " also,", " also ",
            " and do you ", " and what ", " and tell me ", " and can you ",
            " and how do ", " and who ", " and where ",
            " but also ", " but what ", " but can you ",
            ";", " then ",
        ]

        chosen = None
        for sep in separators:
            if sep in lower:
                parts = [x.strip() for x in prompt.split(sep)]
                ml_parts = [x for x in parts if any(k in x.lower() for k in ml_keywords)]
                if ml_parts:
                    chosen = ml_parts[0]
                    break

        if chosen:
            debug["hybrid_detected"] = True
            debug["hybrid_original"] = prompt
            prompt = chosen

        # 4) Rewrite query
        rewritten = self.rewriter_agent.rephrase(conversation, prompt)
        debug["rewritten"] = rewritten
        debug["agent_path"] = "obnoxious -> rewrite"

        # 5) Retrieve
        docs = self.query_agent.query_vector_store(rewritten, k=6)
        debug["docs_count"] = len(docs)
        debug["namespace"] = PINECONE_NAMESPACE
        debug["top_doc_preview"] = (docs[0].page_content[:300] + "...") if docs else ""
        debug["agent_path"] = "obnoxious -> rewrite -> retrieve"

        if not docs:
            debug["agent_path"] = "obnoxious -> rewrite -> retrieve (no_docs)"
            return (
                f"{REFUSAL_PREFIX} irrelevant — No documents were retrieved (check index/namespace).",
                debug,
            )

        # 6) Answer
        answer = self.answering_agent.generate_response(rewritten, docs)
        debug["agent_path"] = "obnoxious -> rewrite -> retrieve -> answer"
        return answer, debug
