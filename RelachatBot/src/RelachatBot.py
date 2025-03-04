import os
from langchain_community.llms.anthropic import Anthropic
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_community.llms.vertexai import VertexAI



class RelachatBot:
    def __init__(self):
        pass

    # Create a new instance of the RelachatBot class
    def main(self):
        print("calling main function")


        model = init_chat_model("gemini-2.0-flash-001", model_provider="google_vertexai")
        messages = [
            SystemMessage("Translate the following from English into Italian"),
            HumanMessage("hi!"),
        ]
        result = model.invoke(messages)
        print(result)
    
if __name__ == "__main__":
    rb = RelachatBot()
    rb.main()
