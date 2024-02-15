"""
Here we follow the outdated but well-explained tutorial https://www.youtube.com/watch?v=aywZrzNaKjs
while replacing tech from OpenAI and PineCone with llama and faiss.
"""

import time
import os

import dotenv

from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains import SimpleSequentialChain
from langchain_community.llms import Ollama
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter



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


if __name__ == "__main__":
    greet()
    pre_flight_checks()
    start = time.process_time()

    chain = build_chain()
    response = ask_prompt(chain)
    splitted = split(response)
    vector_store = store(splitted)
    print(search(vector_store))

    end = time.process_time()
    elapsed = str(round(end - start, 2))
    print(f"Elapsed time: {elapsed} seconds")

    goobye()
