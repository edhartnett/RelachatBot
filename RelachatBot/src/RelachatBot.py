import os
from langchain_community.llms.anthropic import Anthropic
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms.vertexai import VertexAI
from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from GenerateAnalystsState import GenerateAnalystsState
from Perspectives import Perspectives

analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
3. Determine the most interesting themes based upon documents and / or feedback above.
                    
4. Pick the top {max_analysts} themes.

5. Assign one analyst to each theme."""


class RelachatBot:
    def __init__(self):
        pass

    def create_analysts(self, state: GenerateAnalystsState):
        
        """ Create analysts """
        
        topic=state['topic']
        max_analysts=state['max_analysts']
        human_analyst_feedback=state.get('human_analyst_feedback', '')
            
        # Enforce structured output
        structured_llm = self.model.with_structured_output(Perspectives)

        # System message
        system_message = analyst_instructions.format(topic=topic,
                                                                human_analyst_feedback=human_analyst_feedback, 
                                                                max_analysts=max_analysts)

        # Generate question 
        analysts = structured_llm.invoke([SystemMessage(content=system_message)]+[HumanMessage(content="Generate the set of analysts.")])
        
        # Write the list of analysis to state
        return {"analysts": analysts.analysts}

    def human_feedback(self, state: GenerateAnalystsState):
        """ No-op node that should be interrupted on """
        pass

    def should_continue(self, state: GenerateAnalystsState):
        """ Return the next node to execute """

        # Check if human feedback
        human_analyst_feedback=state.get('human_analyst_feedback', None)
        if human_analyst_feedback:
            return "create_analysts"
        
        # Otherwise end
        return END


    # Create a new instance of the RelachatBot class
    def main(self):
        print("calling main function")

        self.model = init_chat_model("anthropic:claude-3-5-sonnet-latest", temperature=0)
        #model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
        messages = [
            SystemMessage("Translate the following from English into Italian"),
            HumanMessage("hi!"),
        ]
        result = self.model.invoke(messages)
        print(result)

        # Add nodes and edges 
        builder = StateGraph(GenerateAnalystsState)
        builder.add_node("create_analysts", self.create_analysts)
        builder.add_node("human_feedback", self.human_feedback)
        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", self.should_continue, ["create_analysts", END])

        # Compile
        memory = MemorySaver()
        graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)

   
if __name__ == "__main__":
    rb = RelachatBot()
    rb.main()
