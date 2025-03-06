from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from Analyst import Analyst

class GenerateAnalystsState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
