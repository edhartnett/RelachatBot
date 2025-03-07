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
from InterviewState import InterviewState
from SearchQuery import SearchQuery
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.document_loaders import WikipediaLoader
import operator
from typing import  Annotated
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from Analyst import Analyst
from langchain_core.messages import get_buffer_string
import operator
from typing import List, Annotated
from typing_extensions import TypedDict
from langgraph.constants import Send


class ResearchGraphState(TypedDict):
    topic: str # Research topic
    max_analysts: int # Number of analysts
    human_analyst_feedback: str # Human feedback
    analysts: List[Analyst] # Analyst asking questions
    sections: Annotated[list, operator.add] # Send() API key
    introduction: str # Introduction for the final report
    content: str # Content for the final report
    conclusion: str # Conclusion for the final report
    final_report: str # Final report

tavily_search = TavilySearchResults(max_results=3)


analyst_instructions="""You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}
        
2. Determine a list of {max_analysts} important themes or questions that should be addressed by the analysts.

3. Examine any editorial feedback that has been optionally provided to guide creation of the analysts: 
        
{human_analyst_feedback}
    
4. Select a list of {max_analysts} famous historical people, like Winston Churchill, Jesus, a Roman Emporer, or American president. 
                    
5. Assign one theme to each analyst."""

# Search query writing
search_instructions = SystemMessage(content=f"""You will be given a conversation between an analyst and an expert. 

Your goal is to generate a well-structured query for use in retrieval and / or web-search related to the conversation.
        
First, analyze the full conversation.

Pay particular attention to the final question posed by the analyst.

Convert this final question into a well-structured web search query""")

question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic. 

Your goal is get funny quotes which are interesting and related to your topic.

Here is your topic of focus and set of goals: {goals}
        
Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.
        
When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

answer_instructions = """You are an expert being interviewed by an analyst.

Here is analyst area of focus: {goals}. 
        
You goal is to answer a question posed by the interviewer.

To answer question, use this context:
        
{context}

When answering questions, follow these guidelines:
        
1. Use only the information provided in the context. 
        
2. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.

3. The context contain sources at the topic of each individual document.

"""

section_writer_instructions = """You are an comedic advice columnist for a newspaper. 
            
Your task is to create a short, easily digestible section of a report based on a set of source documents.

1. Analyze the content of the source documents: 
- The name of each source document is at the start of the document, with the <Document tag.
        
2. Create a report structure using markdown formatting:
- Use ## for the section title
- Use ### for sub-section headers
        
3. Write the report following this structure:
a. Title (## header)
b. Summary (### header)

4. Make your title engaging and funny based upon the focus area of the analyst: 
{focus}

5. For the summary section:
- Set up summary with general background / context related to the focus area of the analyst
- Emphasize what is novel, interesting, or surprising about insights gathered from the interview
- Use direct quotes where appropriate.
- Aim for approximately 400 words maximum
                
6. Final review:
- Ensure the report follows the required structure
- Include no preamble before the title of the report
- Check that all guidelines have been followed"""

report_writer_instructions = """You are a comedic newspaper advice columnists answering a question with help of an expert panel of analysts on this overall topic: 

{topic}
    
You have a team of analysts. Each analyst has done two things: 

1. They conducted an interview with an expert on a specific sub-topic.
2. They write up their finding into a memo.

Your task: 

1. You will be given a collection of memos from your analysts.
2. Think carefully about the insights from each memo.
3. Summarize the conclusions of each analyst serparately. 
4. Inlcude direct quotes from the analyists where appropriate.

To format your report:
 
1. Use markdown formatting. 
2. Include no pre-amble for the report.
3. Use no sub-heading. 
4. Start your report with a single title header: ## Insights

Here are the memos from your analysts to build your report from: 

{context}"""

intro_conclusion_instructions = """You are a comedic newspaper advice columnists answering a question on {topic}

You will be given all of the sections of the report.

You job is to write a crisp and punchy and short introduction or conclusion section.

The user will instruct you whether to write the introduction or conclusion.

Include no pre-amble for either section.

Target around 50 words, using a direct quote from one of the interviews.

Use markdown formatting. 

For your introduction, create a funny title and use the # header for the title.

For your introduction, use ## Introduction as the section header. 

For your conclusion, use ## Conclusion as the section header.

Here are the sections to reflect on for writing: {formatted_str_sections}"""


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


    def search_wikipedia(self, state: InterviewState):   
        """ Retrieve docs from wikipedia """

        # Search query
        structured_llm = self.model.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions]+state['messages'])
        
        # Search
        search_docs = WikipediaLoader(query=search_query.search_query, 
                                    load_max_docs=2).load()

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )

        return {"context": [formatted_search_docs]} 
    
    def search_web(self, state: InterviewState):
        """ Retrieve docs from web search """

        # Search query
        structured_llm = self.model.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions]+state['messages'])
        
        # Search
        search_docs = tavily_search.invoke(search_query.search_query)

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )

        return {"context": [formatted_search_docs]} 

    def generate_question(self, state: InterviewState):
        """ Node to generate a question """

        # Get state
        analyst = state["analyst"]
        messages = state["messages"]

        # Generate question 
        system_message = question_instructions.format(goals=analyst.persona)
        question = self.model.invoke([SystemMessage(content=system_message)]+messages)
            
        # Write messages to state
        return {"messages": [question]}
    
    def search_web(self, state: InterviewState):
    
        """ Retrieve docs from web search """

        # Search query
        structured_llm = self.model.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions]+state['messages'])
        
        # Search
        search_docs = tavily_search.invoke(search_query.search_query)

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
                for doc in search_docs
            ]
        )

        return {"context": [formatted_search_docs]} 

    def search_wikipedia(self, state: InterviewState):
        
        """ Retrieve docs from wikipedia """

        # Search query
        structured_llm = self.model.with_structured_output(SearchQuery)
        search_query = structured_llm.invoke([search_instructions]+state['messages'])
        
        # Search
        search_docs = WikipediaLoader(query=search_query.search_query, 
                                    load_max_docs=2).load()

        # Format
        formatted_search_docs = "\n\n---\n\n".join(
            [
                f'<Document source="{doc.metadata["source"]}" page="{doc.metadata.get("page", "")}"/>\n{doc.page_content}\n</Document>'
                for doc in search_docs
            ]
        )

        return {"context": [formatted_search_docs]} 
    
    def generate_answer(self, state: InterviewState):
        
        """ Node to answer a question """

        # Get state
        analyst = state["analyst"]
        messages = state["messages"]
        context = state["context"]

        # Answer question
        system_message = answer_instructions.format(goals=analyst.persona, context=context)
        answer = self.model.invoke([SystemMessage(content=system_message)]+messages)
                
        # Name the message as coming from the expert
        answer.name = "expert"
        
        # Append it to state
        return {"messages": [answer]}

    def save_interview(self, state: InterviewState):
        
        """ Save interviews """

        # Get messages
        messages = state["messages"]
        
        # Convert interview to a string
        interview = get_buffer_string(messages)
        
        # Save to interviews key
        return {"interview": interview}

    def route_messages(self, state: InterviewState, 
                    name: str = "expert"):

        """ Route between question and answer """
        
        # Get messages
        messages = state["messages"]
        max_num_turns = state.get('max_num_turns',2)

        # Check the number of expert answers 
        num_responses = len(
            [m for m in messages if isinstance(m, AIMessage) and m.name == name]
        )

        # End if expert has answered more than the max turns
        if num_responses >= max_num_turns:
            return 'save_interview'

        # This router is run after each question - answer pair 
        # Get the last question asked to check if it signals the end of discussion
        last_question = messages[-2]
        
        if "Thank you so much for your help" in last_question.content:
            return 'save_interview'
        return "ask_question"
    
    def write_section(self, state: InterviewState):

        """ Node to answer a question """

        # Get state
        interview = state["interview"]
        context = state["context"]
        analyst = state["analyst"]
    
        # Write section using either the gathered source docs from interview (context) or the interview itself (interview)
        system_message = section_writer_instructions.format(focus=analyst.description)
        section = self.model.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Use this source to write your section: {context}")]) 
                    
        # Append it to state
        return {"sections": [section.content]}
    
    def initiate_all_interviews(self, state: ResearchGraphState):
        """ This is the "map" step where we run each interview sub-graph using Send API """    

        # Check if human feedback
        human_analyst_feedback=state.get('human_analyst_feedback')
        if human_analyst_feedback:
            # Return to create_analysts
            return "create_analysts"

        # Otherwise kick off interviews in parallel via Send() API
        else:
            topic = state["topic"]
            return [Send("conduct_interview", {"analyst": analyst,
                                            "messages": [HumanMessage(
                                                content=f"So you said you were writing an article on {topic}?"
                                            )
                                                        ]}) for analyst in state["analysts"]]

    def write_report(self, state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        system_message = report_writer_instructions.format(topic=topic, context=formatted_str_sections)    
        report = self.model.invoke([SystemMessage(content=system_message)]+[HumanMessage(content=f"Write a report based upon these memos.")]) 
        return {"content": report.content}

    def write_introduction(self, state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        
        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        intro = self.model.invoke([instructions]+[HumanMessage(content=f"Write the report introduction")]) 
        return {"introduction": intro.content}

    def write_conclusion(self, state: ResearchGraphState):
        # Full set of sections
        sections = state["sections"]
        topic = state["topic"]

        # Concat all sections together
        formatted_str_sections = "\n\n".join([f"{section}" for section in sections])
        
        # Summarize the sections into a final report
        
        instructions = intro_conclusion_instructions.format(topic=topic, formatted_str_sections=formatted_str_sections)    
        conclusion = self.model.invoke([instructions]+[HumanMessage(content=f"Write the report conclusion")]) 
        return {"conclusion": conclusion.content}

    def finalize_report(self, state: ResearchGraphState):
        """ The is the "reduce" step where we gather all the sections, combine them, and reflect on them to write the intro/conclusion """
        # Save full final report
        content = state["content"]
        if content.startswith("## Insights"):
            content = content.strip("## Insights")
        if "## Sources" in content:
            try:
                content, sources = content.split("\n## Sources\n")
            except:
                sources = None
        else:
            sources = None

        final_report = state["introduction"] + "\n\n---\n\n" + content + "\n\n---\n\n" + state["conclusion"]
        if sources is not None:
            final_report += "\n\n## Sources\n" + sources
        return {"final_report": final_report}

    # Create a new instance of the RelachatBot class
    def main(self):
        print("calling main function")

        self.model = init_chat_model("anthropic:claude-3-5-haiku-latest", temperature=0.7)
        #model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
        #messages = [
        #     SystemMessage("Translate the following from English into Italian"),
        #     HumanMessage("hi!"),
        # ]
        # result = self.model.invoke(messages)
        # print(result)

        # Add nodes and edges 
        interview_builder = StateGraph(InterviewState)
        interview_builder.add_node("ask_question", self.generate_question)
        interview_builder.add_node("search_web", self.search_web)
        interview_builder.add_node("search_wikipedia", self.search_wikipedia)
        interview_builder.add_node("answer_question", self.generate_answer)
        interview_builder.add_node("save_interview", self.save_interview)
        interview_builder.add_node("write_section", self.write_section)

        # Flow
        interview_builder.add_edge(START, "ask_question")
        interview_builder.add_edge("ask_question", "search_web")
        interview_builder.add_edge("ask_question", "search_wikipedia")
        interview_builder.add_edge("search_web", "answer_question")
        interview_builder.add_edge("search_wikipedia", "answer_question")
        interview_builder.add_conditional_edges("answer_question", self.route_messages,['ask_question','save_interview'])
        interview_builder.add_edge("save_interview", "write_section")
        interview_builder.add_edge("write_section", END)

        # Interview 
        memory = MemorySaver()
        #interview_graph = interview_builder.compile(checkpointer=memory).with_config(run_name="Conduct Interviews")

        # # Compile
        # memory = MemorySaver()
        # graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
    # Add nodes and edges 
        builder = StateGraph(ResearchGraphState)
        builder.add_node("create_analysts", self.create_analysts)
        builder.add_node("human_feedback", self.human_feedback)
        builder.add_node("conduct_interview", interview_builder.compile())
        builder.add_node("write_report",self.write_report)
        builder.add_node("write_introduction",self.write_introduction)
        builder.add_node("write_conclusion",self.write_conclusion)
        builder.add_node("finalize_report",self.finalize_report)

        # Logic
        builder.add_edge(START, "create_analysts")
        builder.add_edge("create_analysts", "human_feedback")
        builder.add_conditional_edges("human_feedback", self.initiate_all_interviews, ["create_analysts", "conduct_interview"])
        builder.add_edge("conduct_interview", "write_report")
        builder.add_edge("conduct_interview", "write_introduction")
        builder.add_edge("conduct_interview", "write_conclusion")
        builder.add_edge(["write_conclusion", "write_report", "write_introduction"], "finalize_report")
        builder.add_edge("finalize_report", END)

        # Compile
        #memory = MemorySaver()
        graph = builder.compile(interrupt_before=['human_feedback'], checkpointer=memory)
        
        messages = [HumanMessage(f"My girlfriend hates my cat, what should I do?")]
        #interview = self.interview_graph.invoke({"analyst": analysts[0], "messages": messages, "max_num_turns": 4}, thread)

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
                    print(f"Quirky Fact: {analyst.quirky_fact}")
                    print(f"Relationship History: {analyst.relationship_history}")
                    print("-" * 50)  

        further_feedack = input("Feedback?")
        if (further_feedack == "no"):
            further_feedack = None

        # If we are satisfied, then we simply supply no feedback
        graph.update_state(thread, {"human_analyst_feedback": 
                            further_feedack}, as_node="human_feedback")
        
        # Continue the graph execution to end
        for event in graph.stream(None, thread, stream_mode="updates"):
            print("--Node--")
            node_name = next(iter(event.keys()))
            print(node_name)

        #print(interview)
        final_state = graph.get_state(thread)
        report = final_state.values.get('final_report')
        print(report)

 

if __name__ == "__main__":
    rb = RelachatBot()
    rb.main()
