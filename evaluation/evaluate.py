"""
Evaluation harness: runs all test cases through the multi-agent pipeline
and scores them with the LLM-as-a-Judge.

Usage:
    cd multi-agent-rag-chatbot
    python -m evaluation.evaluate
"""

import os
import json
from collections import defaultdict
from pathlib import Path

from openai import OpenAI

from agents import Head_Agent
from evaluation.judge import LLMJudge


def normalize_bot_output(raw_reply: str) -> str:
    return raw_reply.strip()


def run_single_turn_case(agent: Head_Agent, case: dict):
    conversation = [{"role": "user", "content": case["user_input"]}]
    reply, debug = agent.handle_query(conversation, case["user_input"])
    reply = normalize_bot_output(reply)
    agent_path = "obnoxious->rewrite->retrieve->answer"
    return reply, debug, agent_path


def run_multi_turn_case(agent: Head_Agent, case: dict):
    turns = case["turns"]
    conversation = []
    last_reply = ""
    last_debug = {}

    for t in turns:
        if isinstance(t, dict):
            user_text = t.get("content", "")
        else:
            user_text = str(t)

        conversation.append({"role": "user", "content": user_text})
        reply, debug = agent.handle_query(conversation, user_text)
        reply = normalize_bot_output(reply)
        conversation.append({"role": "assistant", "content": reply})

        last_reply = reply
        last_debug = debug

    agent_path = last_debug.get("agent_path", "unknown")
    return last_reply, last_debug, agent_path


def main():
    openai_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not openai_key or not pinecone_key:
        raise RuntimeError("Missing OPENAI_API_KEY or PINECONE_API_KEY in environment.")

    eval_dir = Path(__file__).parent
    with open(eval_dir / "test_set.json", "r") as f:
        testset = json.load(f)

    cases = testset["cases"]

    agent = Head_Agent(openai_key=openai_key, pinecone_key=pinecone_key)
    judge = LLMJudge(OpenAI(api_key=openai_key))

    scores_by_cat = defaultdict(list)
    failures = []

    for case in cases:
        if case["type"] == "single_turn":
            bot_reply, debug, path = run_single_turn_case(agent, case)
        else:
            bot_reply, debug, path = run_multi_turn_case(agent, case)

        judged = judge.evaluate(case, bot_reply, agent_path=path)
        score = int(judged["score"])
        scores_by_cat[case["category"]].append(score)

        if score == 0:
            failures.append({
                "id": case["id"],
                "category": case["category"],
                "user_input": case.get("user_input", ""),
                "turns": case.get("turns", []),
                "bot_reply": bot_reply,
                "judge": judged,
                "debug": debug,
            })

    overall_scores = [s for arr in scores_by_cat.values() for s in arr]
    overall_acc = sum(overall_scores) / max(1, len(overall_scores))

    print("\n=== Evaluation Report ===")
    for cat, arr in scores_by_cat.items():
        acc = sum(arr) / max(1, len(arr))
        print(f"{cat:12s}: {acc*100:.1f}%  ({sum(arr)}/{len(arr)})")
    print(f"OVERALL     : {overall_acc*100:.1f}%  ({sum(overall_scores)}/{len(overall_scores)})")

    failures_path = eval_dir / "failures.json"
    with open(failures_path, "w") as f:
        json.dump(failures, f, indent=2)

    print(f"\nSaved failure details to {failures_path}")


if __name__ == "__main__":
    main()
