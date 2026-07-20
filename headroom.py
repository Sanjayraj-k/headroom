import json
import os
from dotenv import load_dotenv
from groq import Groq
from headroom import compress

load_dotenv()

client = Groq(
    api_key="#"
)

# ---------------------------------------------------------------------
# Simulate a search tool returning 500 results
# ---------------------------------------------------------------------

search_results = {
    "query": "machine learning tutorials",
    "results": [
        {
            "title": f"Machine Learning Tutorial {i}",
            "url": f"https://example.com/tutorial/{i}",
            "snippet": (
                "This tutorial explains supervised learning, "
                "unsupervised learning, neural networks, decision trees, "
                "random forests, SVMs, gradient boosting, transformers, "
                "LLMs, embeddings, vector databases, and deployment."
            ),
            "score": 100 - (i % 100),
            "author": "OpenAI Research",
            "category": "Machine Learning",
            "tags": [
                "AI",
                "ML",
                "Deep Learning",
                "Python",
                "Tutorial"
            ]
        }
        for i in range(500)
    ]
}

# ---------------------------------------------------------------------
# Agent conversation with a tool call
# ---------------------------------------------------------------------

messages = [
    {
        "role": "system",
        "content": "You are an AI research assistant."
    },

    {
        "role": "user",
        "content": "Search for machine learning tutorials."
    },

    {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search",
                    "arguments": json.dumps(
                        {
                            "query": "machine learning tutorials"
                        }
                    )
                }
            }
        ]
    },

    {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": json.dumps(search_results)
    },

    {
        "role": "user",
        "content": "Give me the five most relevant tutorials."
    }
]

# ---------------------------------------------------------------------
# Compress
# ---------------------------------------------------------------------

result = compress(
    messages,
    model="gpt-4o"
)

print("=" * 60)
print("TOKENS BEFORE :", result.tokens_before)
print("TOKENS AFTER  :", result.tokens_after)
print("TOKENS SAVED  :", result.tokens_saved)
print("RATIO         :", result.compression_ratio)
print("TRANSFORMS    :", result.transforms_applied)
print("=" * 60)

# ---------------------------------------------------------------------
# Send compressed prompt to Groq
# ---------------------------------------------------------------------

response = client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=result.messages,
    temperature=0
)

print(response.choices[0].message.content)
