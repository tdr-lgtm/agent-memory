import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

def get_model(temperature: float = 0.0) -> ChatOpenAI:
    """
    Create a chat model using LLM settings from environment variables.

    Args:
        temperature: Controls randomness in model responses.
            Defaults to 0.0 for more deterministic outputs.

    Returns:
        A configured ChatOpenAI instance.
    """
    return ChatOpenAI(
        model=os.environ["LLM_MODEL"],
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        temperature=temperature,
    )