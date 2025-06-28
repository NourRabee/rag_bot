from langchain_core.prompts import PromptTemplate


def build_chat_prompt(context_docs, current_query):
    context = "\n".join(context_docs) if context_docs else ""

    if not context or not context.strip():
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""Please answer the following question to the best of your knowledge:

    {query}"""
        )
        return prompt_template.format(query=current_query)

    prompt_template = PromptTemplate(
        input_variables=["context", "query"],
        template="""You have access to some context information below. 
        Use it to answer the user's question if it's relevant. If the context is not relevant to the question or doesn't
        contain helpful information, ignore it and answer based on your general knowledge.

    IMPORTANT:
    DON'T MENTION ANYTHING ABOUT IF THE CONTEXT RELEVANT OR NOT WITHIN YOUR RESPONSE.
    If asked whether you remember past messages, politely clarify that you can refer to the current conversation only. 
    Avoid using technical terms like "context window" or saying you don't have memory. 
    Instead, respond in a human-like way, such as: "I can remember what we've talked about here, but not past conversations 
    unless you tell me again."


    Context:
    {context}

    User Question: {query}

    Please provide a helpful and accurate response."""
    )

    formatted_prompt = prompt_template.format(
        context=context,
        query=current_query
    )

    return formatted_prompt
