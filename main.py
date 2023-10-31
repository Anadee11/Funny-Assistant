from dotenv import load_dotenv
import os
from langchain.llms import OpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

load_dotenv()
OPEN_AI_KEY = os.getenv('OPEN_AI_KEY')

llm = OpenAI(openai_api_key=OPEN_AI_KEY)

system_message = "You are a funny assistant. Try responding to each question with a funny answer. Try to avoid answering directly."

template = """Question: {question}

Let's think step by step. What would be a funny take on this?
"""

chat_prompt = ChatPromptTemplate.from_messages([
    ("system",system_message),
    ("human",template)
])

funny_chain = LLMChain(prompt=chat_prompt, llm=llm)

def postprocess(response):
    return response.strip()

def main():
    print("I am active and I try to be funny.")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            print("Funny Assistant: Goodbye! Keep smiling!")
            break

        response = funny_chain.run({"question": user_input})

        response = postprocess(response)
        print(f"Funny Assistant: {response}")

if __name__ == "__main__":
    main()
