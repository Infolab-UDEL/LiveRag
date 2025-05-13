prompt_library = {
    "TEST#dense": {
        "description": "Rewriting queries using LLM.",
        "system_prompt": ("You are an assistant for question-answering tasks."),
        "user_prompt_template": (
            "Delete the premise in the following question , and integrate the words inside the question, answer with only the re write question.\n"
            "Question: {query}\n"
            "Rewrite Question:"
        ),
    },

    "TEST#sparse": {
        "description": "Rewriting queries using LLM.",
        "system_prompt": ("You are an assistant for question-answering tasks."),
        "user_prompt_template": (
            "Provide a better search query for web search engine to answer the given question. Keeping the key terms.\n"
            "Question: {query}.\n"
            "Rewrite Question:"
        ),
    },
    "is_natural": {
        "description": "Rewriting queries using LLM.",
        "system_prompt" : (
                "You are an expert assistant. Use ONLY the context to answer the natural-language question clearly and accurately. "
                "Do not speculate. Keep the answer under 200 tokens."
                "Do not reference nor mention any document number in the answer for example: 'Document5' , 'as described in Document2', 'according to Document4'."
            ),
        "user_prompt_template": "Context:\n{context}\n\nQuestion:\n{question}",
    },

    "is_query_style": {
        "description": "Rewriting queries using LLM.",
        "system_prompt" : (
                "You are an expert assistant. Use ONLY the context to answer the query-style input clearly and accurately. "
                "Do not speculate. Keep the answer under 200 tokens."
                "Do not reference nor mention any document number in the answer for example: 'Document5' , 'as described in Document2', 'according to Document4'."
            ),
        "user_prompt_template": "Context:\n{context}\n\nQuestion:\n{question}",
    }

    # You can add more prompts like this
    # "summarizer": { ... }
}


def build_prompt(prompt_key, **kwargs):
    """
    Returns a list of message dictionaries for the OpenAI API or similar, based on the prompt template.
    Usage: build_prompt("context_selector", context_block="...", question="...")
    """
    template = prompt_library[prompt_key]
    system = template["system_prompt"]
    user = template["user_prompt_template"].format(**kwargs)
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]
