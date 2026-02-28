"""
local_llm.py - Local LLM loader and inference wrapper using Unsloth.
"""


def load_model(model_name: str, max_seq_length: int = 2048):
    """Load a local LLM via Unsloth, optimized for 16GB VRAM."""
    pass


def generate(prompt: str, max_new_tokens: int = 256) -> str:
    """Generate a response from the local LLM for a given prompt."""
    pass


def get_llm():
    """Return a LangChain-compatible LLM wrapper around the local model."""
    pass
