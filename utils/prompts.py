from langchain_core.prompts import PromptTemplate



def build_model_prompt(context_docs, current_query):
    context = "\n".join(context_docs) if context_docs else ""
    # pydantic_parser = PydanticOutputParser(pydantic_object=Output)

    if not context or not context.strip():
        prompt_template = PromptTemplate(
            input_variables=["query"],
            template="""Please answer the following question to the best of your knowledge: {query}"""
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
    
        User Question: {query}"""
    )

    formatted_prompt = prompt_template.format(
        context=context,
        query=current_query,
    )

    return formatted_prompt


def edit_agent_prompt():
    custom_agent_prompt = PromptTemplate(

        input_variables=["input", "agent_scratchpad"],
        template="""
                    You are a helpful assistant with access to tools.

                    Use the following format strictly:

                    Question: {input}
                    Thought: You should always think about what to do.
                    Action: JSON blob with "action" and "action_input" keys.
                    Observation: Tool output here.
                    ... (repeat Thought/Action/Observation as needed)
                    Final Answer: your final response here.

                    If no tool needed, skip Action and Observation and give Final Answer directly.

                    Agent scratchpad:
                    {agent_scratchpad}
                    """
    )
    return custom_agent_prompt
