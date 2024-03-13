import os
import numpy as np
import utils
import streamlit as st
from typing import List
from openai import OpenAI
from langchain.docstore.document import Document
from langchain_community.vectorstores import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser
from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
)
import logging

# for some reason this is needed for chroma to work in deployment
import pysqlite3
import sys

sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

logging.basicConfig(level=logging.INFO)

SEARCH_THRESHOLD = 0.4
QA_THRESHOLD = 0.2  # 0.4
EMBEDDINGS_FUNCTION = "OpenAI"  # "Nomic" or "OpenAI"
LLM_MODEL = "gpt-3.5-turbo"
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

'''
PROMPT = PromptTemplate(
    template="""Dada la siguiente conversación {context} responde a esta nueva pregunta {question}. 
        Si no sabes la respuesta simplemente responde que no sabes la respuesta.""",
    input_variables=["context", "question"],
)
'''

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Eres un útil asistente de IA. Dada una pregunta del usuario y algunos fragmentos de artículos legales, responde a la pregunta del usuario. Es muy importante que estés seguro de la respuesta, si ninguno de los artículos responde claramente a la pregunta, di que no lo sabes.\n\nEstos son los artículos:{context}",
        ),
        ("human", "{question}"),
    ]
)


def create_vector_store(data, embeddings_name, embeddings, path=None):
    """
    Creates a vector store using the given data and embedding function.

    Parameters:
    - data: A list containing the data for creating the vector store.
    - embedding_name: The function used to generate embeddings for the documents.
    - embeddings: The embeddings object.
    - path: The path to save the vector store. If None, the default path is used.

    Returns:
    - vector_store: The vector store object.

    """
    if path is None:
        path = "./vector_stores/chroma_db_" + embeddings_name

    Chroma.from_documents(
        data,
        embeddings,
        persist_directory=path,
    )
    logging.info("Vector store created and saved in " + path)


def get_vector_store(embeddings_name, embeddings, path=None):
    """
    Retrieves a vector store using the given data and embedding function.

    Parameters:
    - embedding_name: The function used to generate embeddings for the documents.
    - embeddings: The embeddings object.
    - path: The path to the vector store. If None, the default path is used.

    Returns:
    - vector_store: The vector store object.

    """
    if path is None:
        path = "./vector_stores/chroma_db_" + embeddings_name
    if os.path.exists(path) and os.listdir(path):
        vector_store = Chroma(persist_directory=path, embedding_function=embeddings)
        logging.info("Vector store loaded from " + path)
    else:
        logging.error("Vector store not found in " + path)
    return vector_store


def filter_by_search(data, search_text, k=100):
    """
    Filter the given data based on a search text and a similarity threshold.

    Args:
        data (pandas.DataFrame): The data to be filtered.
        search_text (str): The text to search for similarity.
        threshold (float, optional): The similarity threshold. Defaults to 0.

    Returns:
        pandas.DataFrame: The filtered data.
    """
    if search_text == "" or search_text is None:
        return data

    else:

        search_text = search_text.lower()

        # check if a month was introduced, this means that the user is looking for a specific month
        # month = utils.get_month(search_text)
        # if month != None:
        #    data_filter_by_month = data[
        #        [month in row for row in data["Norma_translated"]]
        #    ]
        #    # get length of the dataframe
        #    k = len(data_filter_by_month)

        # Now filter based on similarity
        embeddings = get_embeddings(EMBEDDINGS_FUNCTION)
        vector_store = get_vector_store(EMBEDDINGS_FUNCTION, embeddings)

        docs = vector_store.similarity_search_with_score(
            search_text, k
        )  # returns a list of tuples (document, score)

        # get the indexes of the documents that have a similarity score below the threshold
        rows = [doc[0].metadata["page"] for doc in docs if doc[1] < SEARCH_THRESHOLD]

        # if a month was introduced, filter the rows that match the month
        # what we do is getting the concatenation of:
        # - indexes that match the month
        # and
        # - indexes that match the search text
        # append data_filter_by_month.index to the end of the rows list
        # np.append(rows, data_filter_by_month.index)
        # if month != None:
        #    rows = np.intersect1d(rows, data_filter_by_month.index) -> this should keep first the indexes of row but it doesn't

    return data.iloc[rows]


def get_embeddings(type="Nomic"):
    """
    Returns the embeddings object based on the specified type.

    Parameters:
        type (str): The type of embeddings to retrieve. Currently, valid options are "Nomic" and "OpenAI".

    Returns:
        embeddings: The embeddings object based on the specified type. Returns None if the type is not recognized.
    """
    if type == "Nomic":
        from langchain_nomic.embeddings import NomicEmbeddings

        return NomicEmbeddings(model="nomic-embed-text-v1")
    elif type == "OpenAI":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )  # text-embedding-3-small")
    else:
        return None


def get_embedding(text, model="text-embedding-ada-002"):
    text = text.replace("\n", " ")
    return client.embeddings.create(input=[text], model=model).data[0].embedding


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def get_llm(type="gpt-3.5-turbo"):
    """
    Returns an instance of the ChatOpenAI class based on the specified type.

    Parameters:
        type (str): The type of language model to use. Defaults to "gpt-3.5-turbo".

    Returns:
        ChatOpenAI or None: An instance of the ChatOpenAI class based on the specified type,
        None if the type is not recognized.
    """
    if type == "gpt-3.5-turbo":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model_name="gpt-3.5-turbo-0125", temperature=0, streaming=True
        )

    elif type == "gpt-4":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model_name="gpt-4", temperature=0, streaming=True)
    else:
        return None


"""
THIS IS OLD LANGCHAIN, NOW THEY USE PIPES AND RUNNABLES
def create_chain(content):

    # Create embeddings
    # global qa_chain  # No tengo claro esto

    # Add a progress bar or similar

    # Cargar embeddings y crear un vectorstore para utilizar como índice
    embeddings = get_embeddings(EMBEDDINGS_FUNCTION)
    vectorstore = Chroma.from_documents(content, embeddings)
    # crear chain para QA
    llm = get_llm(LLM_MODEL)
    # ConversationalRetrievalChain está construido sobre RetrievalQA
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm,
        vectorstore.as_retriever(),
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": PROMPT},
    )
    print("TODO BIEN")
    st.session_state["qa_chain"] = qa_chain
"""


def format_docs(docs: List[Document]) -> str:
    """Convert Documents to a single string.:"""
    formatted = [
        f"Article Title: {doc.metadata['source']}\nArticle Snippet: {doc.page_content}"
        for doc in docs
    ]
    return "\n\n" + "\n\n".join(formatted) if formatted else "No results"


class Citation(BaseModel):
    source_id: int = Field(
        ...,
        description="The integer ID of a SPECIFIC source which justifies the answer.",
    )
    quote: str = Field(
        ...,
        description="The VERBATIM quote from the specified source that justifies the answer.",
    )


class quoted_answer(BaseModel):
    """Answer the user question based only on the given sources, and cite the sources used."""

    answer: str = Field(
        ...,
        description="The answer to the user question, which is based only on the given sources.",
    )
    citations: List[Citation] = Field(
        ..., description="Citations from the given sources that justify the answer."
    )


def contextualized_chain(llm):
    from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
    from langchain.output_parsers import StrOutputParser

    contextualize_q_system_prompt = """Dado un historial de chat y la última pregunta del usuario, \
    que puede hacer referencia a contenido del historial, formula una pregunta \
    que pueda entenderse sin el historial. NO respondas a la pregunta, \
    solo reformulala si es necesario y devuelvela."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    return contextualize_q_prompt | llm | StrOutputParser()


def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualized_chain()
    else:
        return input["question"]


def create_chain_raw(content):
    """
    Creates a retriever chain and a QA chain for processing content.

    Args:
        content (list): A list of documents to be processed.

    Returns:
        None
    """
    embeddings = get_embeddings(EMBEDDINGS_FUNCTION)
    vectorstore = Chroma.from_documents(content, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 3, "score_threshold": QA_THRESHOLD},
    )

    format = itemgetter("docs") | RunnableLambda(format_docs)
    retriever_chain = RunnableParallel(
        question=RunnablePassthrough(), docs=retriever
    ).assign(
        context=format
    )  # | itemgetter("context")
    st.session_state["retriever_chain"] = retriever_chain
    logging.info("Retriever chain created.")

    llm = get_llm(LLM_MODEL)
    llm = llm.bind_tools(
        [quoted_answer],
        tool_choice="quoted_answer",
    )

    output_parser = JsonOutputKeyToolsParser(
        key_name="quoted_answer", return_single=True
    )
    answer_chain = PROMPT | llm | output_parser
    qa_chain = (
        RunnableParallel(question=RunnablePassthrough(), context=RunnablePassthrough())
        .assign(answer=answer_chain)
        .pick(["answer", "docs"])
    )
    st.session_state["qa_chain"] = qa_chain
    logging.info("QA chain created.")


def format_response(data, docs, response):
    # line breaks are done with 2 spaces + \n
    # display answer
    answer = "  \n  \nRespuesta extraída de "

    # display citations
    for i in range(len(response["answer"]["citations"])):
        doc_info = docs[response["answer"]["citations"][i]["source_id"] - 1]
        doc = doc_info.metadata["source"]
        # redo the original name - undo the changes made in the download_pdfs function
        doc = doc.split("/")[1].replace("_", " ").replace("-", "/").split(".pdf")[0]
        doc_url = data[data["Norma"] == doc]["URL"].to_list()[0]

        answer += (
            "["
            + doc
            + "]("
            + doc_url
            + "),"
            + " página "
            + str(doc_info.metadata["page"])
            + ":  \n*..."
            + utils.convert_to_utf8(response["answer"]["citations"][i]["quote"])
            + "...*  \n\n"
        )

    return answer


def quote_in_context(qa_chain_output, retriever_chain_output):
    """
    Check if return quote by the llm is present in the context.
    If the quote is not present in the context, the answer should not be valid.

    Args:
        qa_chain_output (dict): The output from the QA chain.
        retriever_chain_output (dict): The output from the retriever chain.

    Returns:
        bool: True if the quote is present in the context, False otherwise.
    """
    # [:-1] to remove the last character which is '.' and replace " " by "" to remove blank spaces
    logging.info("\nquote_in_context?\n")
    quote = (
        utils.convert_to_utf8(qa_chain_output["answer"]["citations"][0]["quote"])[:-1]
        .replace("\n", "")
        .replace(" ", "")
        .replace("\xad", "")
    )
    context = retriever_chain_output["context"].replace("\n", "").replace(" ", "")
    logging.info("\n" + quote)
    logging.info("\n" + context)

    return quote in context


def generate_response(data, user_prompt):
    retriever_chain = st.session_state["retriever_chain"]
    qa_chain = st.session_state["qa_chain"]

    logging.info("Calling the retriever chain...")
    retriever_chain_output = retriever_chain.invoke(user_prompt)
    logging.info("Retriever chain output: \n" + str(retriever_chain_output))
    # context = retriever_chain_output | itemgetter("context")
    context = retriever_chain_output["context"]
    if context == "No results":
        answer = "Lo siento, no he encontrado nada relacionado con tu consulta."
    else:
        try:
            logging.info("Calling the QA chain...")
            qa_chain_output = qa_chain.invoke(
                {"question": user_prompt, "context": context}
            )
            answer = utils.convert_to_utf8(qa_chain_output["answer"]["answer"])
            logging.info("QA chain output: \n" + str(qa_chain_output))

            # answer found in the context
            if len(qa_chain_output["answer"]["citations"]) > 0 and quote_in_context(
                qa_chain_output, retriever_chain_output
            ):
                answer += format_response(
                    data, retriever_chain_output["docs"], qa_chain_output
                )
            # else:
            # the model has not found an answer, it would likely say something like "I don't know"
            # answer += qa_chain_output["answer"]["answer"]
            # answer = "No he encontrado nada relacionado con tu consulta."
        except Exception as e:
            logging.error("Error en la respuesta: ", e)
            answer = "Lo siento, ha ocurrido un error inesperado."

    st.session_state.messages.append({"role": "assistant", "content": answer})
    return answer


# --------------------------------------------------------------
