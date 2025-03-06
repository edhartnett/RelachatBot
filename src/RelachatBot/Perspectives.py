from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from Analyst import Analyst 

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )
