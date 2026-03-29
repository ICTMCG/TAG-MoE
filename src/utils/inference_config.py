import random


DEFAULT_WIDTH = 1024
DEFAULT_HEIGHT = 1024
DEFAULT_SEED = -1
DEFAULT_TRUE_CFG_SCALE = 4.0
DEFAULT_NUM_INFERENCE_STEPS = 30
DEFAULT_NEGATIVE_PROMPT = ""


def normalize_negative_prompt(value: str | None) -> str:
    if value is None or not str(value).strip():
        return " "
    return str(value)


def generate_random_seed() -> int:
    return random.randint(0, 2**32 - 1)
