import knapsack
from workflow.base import settings
from typing import Tuple

def construct(instructions:str, *args)->str:
    context_length = settings.context_length