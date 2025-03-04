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
        
2. Determine a list of important themes or questions that should be addressed by the analysts.

3. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
4. Select a list of people, alive or historical, based upon documents and / or feedback above. Include some famous 
people, like Winston Churchill, Jesus, or Marie Curie. Also include some random people from history, like Johm Carpenter, 
a 12th century builder from London, or Marcus Lepitus, a Roman soldier. 
                    
5. Pick the top {max_analysts} people to be analysts.

6. Assign one analyst to each theme."""


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

        # Input
        max_analysts = 3 
        topic = '''
        Relationship advice for dealing with difficult problems with:
         * friends
         * romantic partners
         * roommates and housemates
         * parents
         * siblings
         * business associates
         * strangers
        '''
        thread = {"configurable": {"thread_id": "1"}}

        # Run the graph until the first interruption
        for event in graph.stream({"topic":topic,"max_analysts":max_analysts,}, thread, stream_mode="values"):
            # Review
            analysts = event.get('analysts', '')
            if analysts:
                for analyst in analysts:
                    print(f"Name: {analyst.name}")
                    print(f"Affiliation: {analyst.affiliation}")
                    print(f"Role: {analyst.role}")
                    print(f"Description: {analyst.description}")
                    print("-" * 50)  

        # If we are satisfied, then we simply supply no feedback
        further_feedack = None
        graph.update_state(thread, {"human_analyst_feedback": 
                            further_feedack}, as_node="human_feedback")
   
if __name__ == "__main__":
    rb = RelachatBot()
    rb.main()
