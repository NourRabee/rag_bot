def build_prompt(context_docs, current_query):
    context = "\n".join(context_docs)
    return (
        f"{context}\n\nUser: {current_query}"
    )
