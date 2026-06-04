import os
from dotenv import load_dotenv

load_dotenv()
from ragas.llms import LangchainLLMWrapper

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

from langchain_openai import ChatOpenAI


PROVIDERS = [

    LangchainLLMWrapper(
        ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0,
        )
    ),

    LangchainLLMWrapper(
        ChatGroq(
            model_name="llama-3.3-70b-versatile",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0,
        )
    ),

    LangchainLLMWrapper(
        ChatOpenAI(
            model="qwen/qwen3-32b",
            api_key=os.getenv("OPENROUTER_API_KEY"),
            base_url="https://openrouter.ai/api/v1",
            temperature=0,
        )
    ),
]

_idx = 0


def get_ragas_llm():
    global _idx

    llm = PROVIDERS[_idx]

    _idx = (_idx + 1) % len(PROVIDERS)

    return llm