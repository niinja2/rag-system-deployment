from pathlib import Path


def load_system_prompt(path="prompts/rag_system_prompt.txt"):
    return Path(path).read_text(encoding="utf-8").strip()


def build_prompt(query, chunks):

    context_blocks = []

    for chunk in chunks:
        context_blocks.append(
            f"[{chunk['chunk_id']}]\n{chunk['text']}"
        )

    context = "\n\n".join(context_blocks)

    messages = [
        {
            "role": "system",
            "content": load_system_prompt()
        },
        {
            "role": "user",
            "content": f"""
            
                Question:
                {query}

                Context:
                {context}
                """
        }
    ]

    return messages
