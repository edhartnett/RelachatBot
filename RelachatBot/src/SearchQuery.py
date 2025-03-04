import operator
from typing import  Annotated
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from Analyst import Analyst

class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search query for retrieval.")