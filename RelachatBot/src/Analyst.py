from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

class Analyst(BaseModel):
    affiliation: str = Field(
        description="Primary affiliation of the analyst.",
    )
    name: str = Field(
        description="Name of the analyst, dates of birth/death if applicable.",
    )
    role: str = Field(
        description="Analysts role, e.g. historian, journalist, etc.",
    )
    description: str = Field(
        description="Description of the analyst focus, concerns, and motives.",
    )
    quirky_fact: str = Field(
        description="A quirky fact about the analyst; something funny which has nothing to do with relationships.",
    )
    relationship_history: str = Field(
        description="The relationship history of the analyst, including marriages, children, and other significant relationships.",
    )

    @property
    def persona(self) -> str:
        return f'''Name: {self.name}\nRole: {self.role}\nAffiliation: {self.affiliation}\n
        Description: {self.description}\nQuirky Fact: {self.quirky_fact}\nRelationship History: {self.relationship_history}'''
