"""
Here we follow the outdated but well-explained tutorial https://www.youtube.com/watch?v=aywZrzNaKjs
while replacing tech from OpenAI and PineCone with llama and faiss.
"""

import time
import os

import dotenv

from agents import AlpacaLLM

from langchain import PromptTemplate
from langchain import hub
from langchain.agents import Tool
from langchain.agents import AgentExecutor
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.chat_models import ChatOllama
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_experimental.utilities import PythonREPL


def greet():
    print("ohai! 🦈")


def pre_flight_checks():
    env = dotenv.find_dotenv()
    print(f"dotenv file: >>{env}<<")
    dotenv.load_dotenv(env)


def goobye():
    print("k thx bye! 👋")


def build_chain():
    model = os.environ.get("OLLAMA_MODEL")

    llm = Ollama(model=model)
    prompt = PromptTemplate(
        input_variables=["llm_persona"],
        template="You are {llm_persona}. Please explain the concept of a large language model."
    )

    expert_chain = LLMChain(
        llm=llm,
        prompt=prompt
    )

    eli5_chain = LLMChain(
        llm=llm,
        prompt=PromptTemplate(
            input_variables=["llm_persona"],
            template="You are {llm_persona}. Now please explain this like I am five years old in 500 words."
        )
    )

    return SimpleSequentialChain(
        chains=[expert_chain, eli5_chain],
        verbose=True
    )


def ask_prompt(sequential_chain):
    llm_persona = "an expert data scientist"

    print("===== PROMPT =====")
    for c in sequential_chain.chains:
        print(c.prompt)
    print(f"llm_persona = {llm_persona}")

    print("\n===== RESPONSE =====")
    response = sequential_chain.run(llm_persona)
    print(response)
    print("===== END OF RESPONSE =====\n")

    return response


def split(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=100,
        chunk_overlap=0
    )
    return text_splitter.create_documents([text])


def store(splitted):
    embeddings_model = os.environ.get("HUGGINGFACE_EMBEDDING_MODEL")
    model_kwargs = {'device': 'cuda'}
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs=model_kwargs)

    vector_store = FAISS.from_documents(splitted, embeddings)

    return vector_store


def search(vector_store):
    print("===== SEARCH =====")
    query = "How can large language models cause trouble?"
    print(query)
    return vector_store.similarity_search(query)


# def create_agent():
#     python_repl = PythonREPL()
#     repl_tool = Tool(
#         name="python_repl",
#         description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
#         func=python_repl.run
#     )
#     agent_tools = [repl_tool]
#
#     instructions = """You are an agent designed to write and execute python code to answer questions.
#     You have access to a python REPL, which you can use to execute python code.
#     If you get an error, debug your code and try again.
#     Only use the output of your code to answer the question.
#     You might know the answer without running any code, but you should still run the code to get the answer.
#     If it does not seem like you can write code to answer the question, just return "I don't know" as the answer.
#     """
#     base_prompt = hub.pull("langchain-ai/react-agent-template")
#     prompt = base_prompt.partial(instructions=instructions)
#     agent = create_(ChatOllama(temperature=0), agent_tools, prompt)
#     return AgentExecutor(agent=agent, tools=agent_tools, verbose=True)


if __name__ == "__main__":
    greet()
    pre_flight_checks()
    start = time.process_time()

    chain = build_chain()
    response = ask_prompt(chain)
    splitted = split(response)
    vector_store = store(splitted)
    print(search(vector_store))

    # agent = create_agent()
    # agent.invoke({"input": "What is the 10th fibonacci number?"})

    end = time.process_time()
    elapsed = str(round(end - start, 2))
    print(f"Elapsed time: {elapsed} seconds")

    goobye()
