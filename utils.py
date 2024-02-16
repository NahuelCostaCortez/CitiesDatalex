import os
import pandas as pd
import numpy as np
import streamlit as st
import requests
import logging
logging.basicConfig(level=logging.INFO)
from typing import List


from operator import itemgetter
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate

from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda
)

# will simply take the input and pass it through
from langchain.output_parsers.openai_tools import JsonOutputKeyToolsParser

os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]
os.environ['NOMIC_API_KEY'] = st.secrets["NOMIC_API_KEY"]

#from langchain.chains import ConversationalRetrievalChain
# from langchain_community.llms import OpenAI

FOLDER_PATH = "documents"
SEARCH_THRESHOLD = 0.88
QA_THRESHOLD = 0.4
EMBEDDINGS_FUNCTION = "OpenAI" # "Nomic" or "OpenAI"
LLM_MODEL = "gpt-3.5-turbo"

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
            "Eres un útil asistente de IA. Dada una pregunta del usuario y algunos fragmentos de artículos legales, responde a la pregunta del usuario. Si ninguno de los artículos responde a la pregunta, di que no lo sabes.\n\nEstos son los artículos:{context}",
        ),
        ("human", "{question}"),
    ]
)


# ---------------------------- DATA ----------------------------
@st.cache_data  # for not loading the data every time the page is refreshed
def load_data():
    """
    Load data from an Excel file.

    Returns:
        df_data (pandas.DataFrame): The data from the 'Registros' sheet.
        df_tesauro (pandas.DataFrame): The data from the 'Tesauro' sheet.
    """
    logging.info("Loading data...")

    # Sheet with norms
    df_data = pd.read_csv("./data/data.csv")

    # Remove rows with erroneous URLs
    with open("./data/erroneous_urls.txt", "r") as file:
        erroneous_urls = file.readlines()

    indexes_to_remove = []
    for url in erroneous_urls:
        url = url.strip("\n")
        indexes = df_data[df_data["URL"] == url].index.to_list()
        indexes_to_remove.extend(indexes)

    # Sheet with tesauro
    df_data = df_data.drop(indexes_to_remove)
    df_tesauro = pd.read_excel(
        "./data/Datos_DL_2022_Final.xlsm", engine="openpyxl", sheet_name="Tesauro"
    )

    logging.info("Data loaded.")

    return df_data, df_tesauro


def download_pdfs(names, urls):
    """
    Download PDF files from URLs and save them with corresponding names.

    Args:
        names (list): List of names to use for renaming downloaded files.
        urls (list): List of URLs of the PDF files to download.

    Returns:
        pdf_names (list): List of names of the downloaded PDF files.
    """

    logging.info("Downloading PDFs...")

    urls = rename_files(urls)

    pdf_names = []
    # Iterate over names and urls simultaneously
    for name, url in zip(names, urls):
        mod_name = name.replace(" ", "_")
        mod_name = mod_name.replace("/", "-")

        # Check if PDF file already exists
        if os.path.exists(os.path.join(FOLDER_PATH, f"{mod_name}.pdf")):
            print(f"PDF for '{name}' already exists")
            pdf_names.append(name)
            continue
        # Send request to download PDF file
        else:
            print(f"Downloading PDF for '{name}' from URL: {url}")
            try:
                # Hardcoded
                if (
                    url
                    != "https://www.boe.es/buscar/pdf/2015/BOE-A-2015-10202-consolidado.pdf"
                    and url
                    != "https://sede.madrid.es/portal/site/tramites/menuitem.5dd4485239c96e10f7a72106a8a409a0/?vgnextoid=3637f58062f0d710VgnVCM1000001d4a900aRCRD&vgnextchannel=e81965dd72ede410VgnVCM1000000b205a0aRCRD&vgnextfmt=default"
                ):
                    response = requests.get(url)

                    # Check if request was successful
                    if response.status_code == 200:
                        # Save the downloaded PDF with corresponding name in the specified folder path
                        with open(
                            os.path.join(FOLDER_PATH, f"{mod_name}.pdf"), "wb"
                        ) as file:
                            file.write(response.content)
                        logging.info(
                            f"Downloaded PDF for '{name}' from URL: {url} and saved as '{os.path.join(FOLDER_PATH, f'{mod_name}.pdf')}'"
                        )
                        pdf_names.append(name)
                    else:
                        logging.error(
                            "Error descargando el contenido de ",
                            name,
                            ". Inténtelo más tarde.",
                        )
                else:
                    pdf_names.append(name)
            except Exception as e:
                logging.error("Error descargando el contenido de ", name, ". Traceback: ", e)

    return pdf_names


def rename_files(urls):
    """
    Renames the files in the given list of URLs based on specific conditions.

    Args:
        urls (list): A list of URLs.

    Returns:
        list: The modified list of URLs with renamed files.
    """
    for i in range(len(urls)):
        # for those urls that start with "https://eur-lex.europa" we need to change the substring "/EN/TXT" for "/ES/TXT/PDF"
        if urls[i].startswith("https://eur-lex.europa"):
            urls[i] = urls[i].replace("/EN/TXT", "/ES/TXT/PDF")

        # for those urls that start with "https://www.boe.es" and do not end in "pdf" we need to change the substring "act.php?id=" for pdf/2021/
        # and add "-consolidado.pdf" at the end
        if (
            urls[i].startswith("https://www.boe.es")
            and not urls[i].endswith("pdf")
            and not "codigo" in urls[i]
        ):
            # get the year from the url
            # print(i)
            year = urls[i].split("-")[-2]
            urls[i] = (
                urls[i].replace("act.php?id=", "pdf/" + year + "/") + "-consolidado.pdf"
            )

        # for those urls that contain "codigo" replace the substring "codigo.php?id=" by "abrir_pdf.php?fich="
        # and replace everything that comes after a "&" by ".pdf"
        if "codigo" in urls[i] and not "nota" in urls[i]:
            urls[i] = urls[i].replace("codigo.php?id=", "abrir_pdf.php?fich=")
            urls[i] = urls[i].split("&")[0] + ".pdf"

    return urls


def extract_text_from_pdf(pdf_names):

    logging.info("Extracting text from PDFs...")

    # user has uploaded a single pdf
    if type(pdf_names) != list:
        loader = PyPDFLoader(pdf_names)
        content = loader.load_and_split()

    # user has selected/downloaded multiple pdfs
    else:
        content = []
        # Cargar documento/s
        for index, pdf_name in enumerate(pdf_names):
            pdf_name = pdf_name.replace(" ", "_")
            pdf_name = pdf_name.replace("/", "-")
            pdf_name = os.path.join(FOLDER_PATH, f"{pdf_name}.pdf")
            loader = PyPDFLoader(pdf_name)
            pages = loader.load_and_split()
            if index == 0:
                content = pages
            else:
                content.append(pages)

        # Vaciar directorio de documentos
        # for file in os.listdir(FOLDER_PATH):
        #    os.remove(os.path.join(FOLDER_PATH, file))
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0, separator="\n")
    content = text_splitter.split_documents(content)
    
    logging.info("Text extracted from PDFs.")

    return content


# -------------------------- FILTERS ---------------------------
def ambito_tematico_options():
    return [
        "Sostenibilidad Económica",
        "Sostenibilidad Ambiental",
        "Cambio Climático",
        "Sostenibilidad Social",
        "Gobernanza Urbana",
    ]


def get_materia_options(selected_ambito_tematico):
    if selected_ambito_tematico == "Sostenibilidad Económica":
        materia_options = [
            "Competitividad",
            "Economía colaborativa",
            "Economía informal",
            "Economía/desarrollo local",
            "Economía del conocimiento",
            "Oferta/calidad de empleo",
            "Regulación de mercados",
            "Localización de actividades económicas",
            "Estructura empresarial/emprendimiento",
            "Marca ciudad",
            "Industrias culturales",
            "Smart region/Estrategias de especialización inteligente",
            "Smart city",
        ]
    elif selected_ambito_tematico == "Sostenibilidad Ambiental":
        materia_options = [
            "Medio ambiente natural",
            "Medio ambiente urbano",
            "Prevención/gestión riesgos accidentes industriales",
            "Prevención/gestión desastres naturales",
            "Eficiencia en el uso de recursos naturales",
        ]
    elif selected_ambito_tematico == "Cambio Climático":
        materia_options = [
            "Energía",
            "Movilidad urbana",
            "Emisiones de gases de efecto invernadero (GEI)",
            "Efectos peligrosos del cambio climático",
            "Mitigación de los efectos del cambio climático",
            "Adaptación a los efectos del cambio climático",
            "Geoingeniería",
        ]
    elif selected_ambito_tematico == "Sostenibilidad Social":
        materia_options = [
            "Género",
            "Juventud",
            "Población",
            "Igualdad/no discriminación",
            "Servicios básicos",
            "Segregación urbana",
            "Vulnerabilidad social",
            "Innovación social",
        ]
    elif selected_ambito_tematico == "Gobernanza Urbana":
        materia_options = [
            "Competencias",
            "Escalas territoriales",
            "Participación ciudadana y de agentes sociales",
            "Planeamiento urbano",
            "Cooperación territorial (Cross border)",
            "Financiación",
            "Regulación",
        ]
    else:
        materia_options = ["Cualquiera"]
    return materia_options


def get_submateria(df, ambito, materia, nivel):
    column = ambito + "_nivel" + str(nivel)
    index_column = df.columns.get_loc(column)

    row = np.where(df[column] == materia)[0][0]

    submaterias = []

    # while in the previous column there is a nan value iterate
    # Example:
    #  Sostenibilidad Económica  | Sostenibilidad Económica_nivel1
    #  Economía del conocimiento |
    #                            | Investigación e innovación
    #  Oferta/calidad de empleo  |
    while row + 1 < df.shape[0] and pd.isna(df.iloc[row + 1, index_column]):
        content = df.iloc[row + 1, index_column + 1]
        if pd.notna(content):
            submaterias.append(content)
        row += 1

    return submaterias


def reset_filters():
    st.session_state.ambito_tematico = "Cualquiera"
    st.session_state.materia = "Cualquiera"
    st.session_state.comunidad = "Cualquiera"
    st.session_state.municipio = "Cualquiera"
    st.session_state.input_keywords = ""
    st.session_state["checkbox_values"] = False


# --------------------------------------------------------------


# ------------------------  Search ------------------------ #
def search_logic(
    df_data,
    df_tesauro,
    filters,
    search_text,
    selected_ambito_tematico,
    selected_materia,
    selected_submaterias,
    selected_comunidad,
    selected_municipio,
    input_keywords,
):

    filtered_rows = None

    if filters:
        if search_text == "" or search_text is None:
            if (
                selected_ambito_tematico == "Cualquiera"
                and selected_materia == "Cualquiera"
                or selected_materia == "Cualquiera"
            ):
                st.warning(
                    "Selecciona al menos un ámbito temático y una materia para realizar la búsqueda"
                )
                st.stop()

            else:
                
                filtered_rows = filter_by_filters(
                    df_data,
                    df_tesauro,
                    input_keywords,
                    selected_ambito_tematico,
                    selected_materia,
                    selected_submaterias,
                    selected_comunidad,
                    selected_municipio,
                )

        else:
            filtered_rows = filter_by_search(df_data, search_text)
            filtered_rows = filter_by_filters(
                filtered_rows,
                df_tesauro,
                input_keywords,
                selected_ambito_tematico,
                selected_materia,
                selected_submaterias,
                selected_comunidad,
                selected_municipio,
            )

    else:
        if search_text == "" or search_text is None:
            pass
        else:
            logging.info("Filtering by search text...")
            filtered_rows = filter_by_search(df_data, search_text)

    # No results
    if filtered_rows is None or filtered_rows.empty:
        st.write("No hay resultados para los filtros seleccionados")
    else:
        display_results(filtered_rows, df_tesauro)


def get_codes(df_tesauro, row, column, selected_submaterias):
    """
    Retrieves the codes associated with the selected submaterias from the given dataframe.

    Parameters:
    - df_tesauro (pandas.DataFrame): The dataframe containing the tesauro data.
    - row (int): The row index of the submateria.
    - column (int): The column index of the codes.
    - selected_submaterias (list): The list of selected submaterias.

    Returns:
    - codes (list): The list of codes associated with the selected submaterias.
    """
    codes = []
    # if there are submaterias selected
    if selected_submaterias != []:
        # get codes of the selected submaterias
        for submateria in selected_submaterias:
            codes.append(
                df_tesauro.iloc[
                    np.where(df_tesauro.iloc[:, column + 1] == submateria)[0][0],
                    column - 1,
                ]
            )
    # No submaterias selected
    else:
        codes = [df_tesauro.iloc[row, column - 1]]
    return codes


def get_rows_from_codes(df, ambito, codes):
    """
    Returns the unique rows indices from a DataFrame `df` based on the given `ambito` and `codes`.

    Parameters:
        df (pandas.DataFrame): The DataFrame to search in.
        ambito (str): The column name in `df` to search for codes.
        codes (list): The list of codes to search for.

    Returns:
        numpy.ndarray: The unique row indices that match the given `ambito` and `codes`.
    """
    return np.unique(
        np.concatenate(
            [
                np.where([code in row.split("; ") for row in df[ambito + ".1"]])[0]
                for code in codes
            ]
        )
    )


def get_vector_store(data, embeddings_function, embeddings, path=None):
    """
    Retrieves or creates a vector store using the given data and embedding function.

    Parameters:
    - data: A list containing the data for creating the vector store.
    - embedding_function: The function used to generate embeddings for the documents.

    Returns:
    - vector_store: The vector store object.

    """
    if path is None:
        path = "./vector_stores/chroma_db_" + embeddings_function
    if os.path.exists(path) and os.listdir(path):
        vector_store = Chroma(persist_directory=path, embedding_function=embeddings)
        logging.info("Vector store loaded from " + path)
    # if path is empty, create the retriever
    else:
        vector_store = Chroma.from_documents(
            data,
            embeddings,
            persist_directory=path,
        )
        logging.info("Vector store created and saved in " + path)
    return vector_store


def filter_by_search(data, search_text):
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
        embeddings = get_embeddings(EMBEDDINGS_FUNCTION)

        # convert normas to documents
        documents = [
            Document(
                page_content=norm,  # assuming 'title' is a field in each norm
                metadata={"source": "Normas", "page": i},
            )
            for i, norm in enumerate(data["Norma_translated"])
        ]

        vector_store = get_vector_store(documents, EMBEDDINGS_FUNCTION, embeddings)
        # retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 5})
        # docs = retriever.get_relevant_documents(search_text)
        docs = vector_store.similarity_search_with_score(
            search_text, k=30
        )  # returns a list of tuples (document, score)
        rows = np.unique(
            [doc[0].metadata["page"] for doc in docs if doc[1] < SEARCH_THRESHOLD]
        )
        data = data.iloc[rows]

        logging.info("Data filtered by search text, results found: " + str(len(data)) + " rows.")

    return data


def filter_by_filters(
    df_data,
    df_tesauro,
    input_keywords,
    selected_ambito_tematico,
    selected_materia,
    selected_submaterias,
    selected_comunidad,
    selected_municipio,
):
    
    filtered_rows = df_data
    if selected_ambito_tematico != "Cualquiera":
        # Filter by selected ambito tematico (last columns of the 'Registros' sheet)
        filtered_rows = filtered_rows[pd.notna(filtered_rows[selected_ambito_tematico + ".1"])]

    if selected_materia != "Cualquiera":
        # Filter by selected materia
        # 1. Get the column index of the selected materia in the 'tesauro' sheet
        column = df_tesauro.columns.get_loc(selected_ambito_tematico) + 1
        # 2. Get the row that matches the selected materia
        row = np.where(df_tesauro.iloc[:, column] == selected_materia)[0][0]
        # 3. Get the code/s of the selected materia or submateria (the code is in the previous column)
        codes = get_codes(df_tesauro, row, column, selected_submaterias)
        # 4. Now we have the code/s of the selected materia, we can filter the rows that match the selected materia
        filtered_materia = get_rows_from_codes(
            filtered_rows, selected_ambito_tematico, codes
        )
        filtered_rows = filtered_rows.iloc[filtered_materia]

    # Filter by comunidad autonoma
    if selected_comunidad != "Cualquiera":
        filtered_rows = filtered_rows[filtered_rows["CCAA"] == selected_comunidad]

    # Filter by municipio
    if selected_municipio != "Cualquiera":
        filtered_rows = filtered_rows[filtered_rows["Ciudad"] == selected_municipio]

    # Filter by keywords
    if input_keywords:
        keywords = input_keywords.split(";")
        for i, keyword in enumerate(keywords):
            filter_keyword = filtered_rows[
                filtered_rows["Norma"].str.contains(keyword, case=False)
            ]
            if i == 0:
                filtered_rows = filter_keyword
            # Before concatenating, remove the rows that are already in filtered_rows
            filter_keyword = filter_keyword[
                ~filter_keyword["Norma"].isin(filtered_rows["Norma"])
            ]
            filtered_rows = pd.concat([filtered_rows, filter_keyword])

    return filtered_rows


def display_results(filtered_rows, df_tesauro):
    # Display results
    st.markdown("### Resultados")

    colms = st.columns((2, 1, 1, 1, 1))
    fields = ["Norma", "CCAA", "Ciudad", "Descriptores", "Acción"]
    # Display headers in bold
    for col, field_name in zip(colms, fields):
        col.markdown("**" + field_name + "**")

    # Form is necessary to keep the checkboxes checked when reloading the page
    checkbox_statusses = []
    with st.form("results", clear_on_submit=False):
        for index, row in filtered_rows.iterrows():
            col1, col2, col3, col4, col5 = st.columns((2, 1, 1, 1, 1))

            # Norma
            col1.markdown(
                "[{0}]({1})".format(row["Norma"], row["URL"]),
                unsafe_allow_html=True,
            )
            # CCAA
            if pd.isna(row["CCAA"]):
                col2.write("No aplica")
            else:
                col2.write(row["CCAA"])
            # Ciudad
            if pd.isna(row["Ciudad"]):
                col3.write("No aplica")
            else:
                col3.write(row["Ciudad"])

            # Descriptores
            labels = []
            # Check all the ambitos tematicos because the descriptores can be in any of them
            for ambito in ambito_tematico_options():
                if pd.notna(row[ambito + ".1"]):
                    for code in row[ambito + ".1"].split(";"):
                        if len(code) > 1:
                            # remove blank spaces
                            code = code.strip()
                            # get the column and row of the code
                            code_column = df_tesauro.columns.get_loc(ambito)
                            # get the row that matches the selected ambito
                            code_row = np.where(
                                df_tesauro.iloc[:, code_column] == code
                            )[0][0]
                            # the name of the label should be to the right
                            code_column += 1
                            while pd.isna(df_tesauro.iloc[code_row, code_column]):
                                code_column += 1
                            labels.append(df_tesauro.iloc[code_row, code_column])

            if len(labels) > 0:
                col4.write(", ".join(labels))

            # Acción
            checkbox_statusses.append(
                col5.checkbox(
                    "Añadir",
                    key=index,
                    value=st.session_state["checkbox_values"],
                )
            )

        # Add a checkbox to select all the rows
        checkbox_all = st.checkbox("Añadir todos", key="all")

        # Add a button to submit the form
        submitted = st.form_submit_button(
            "Cargar documento/s",
            help="Se proporcionarán los documentos seleccionados al asistente",
        )
        # when pressing submitted get the values of the checkboxes
        if submitted:
            # all selected
            if checkbox_all:
                checkbox_statusses = [True for i in range(len(filtered_rows))]
                # when pressing the button, the value of all checkboxes is set to True
                st.session_state["checkbox_values"] = True
            # "Añadir todos" is not selected
            else:
                # when pressing the button, the value of all checkboxes is set to False
                st.session_state["checkbox_values"] = False

            # if more than 5 checkboxes are selected, show a warning
            if len(np.where(checkbox_statusses)[0]) > 5:
                st.warning(
                    "El número máximo de documentos que se pueden cargar actualmente es 5."
                )
                return
            # if no checkboxes are selected, show a warning
            if len(np.where(checkbox_statusses)[0]) == 0:
                st.warning("Selecciona al menos un documento.")
                return

            # filter the rows that have been checked
            filtered_checked_rows = filtered_rows[checkbox_statusses]

            # Download the pdfs based on the URLs
            available_pdfs = download_pdfs(
                filtered_checked_rows["Norma"],
                filtered_checked_rows["URL"].to_list(),
            )

            for pdf_name in filtered_checked_rows["Norma"]:
                if pdf_name not in available_pdfs:
                    # escribir mensaje de error en rojo
                    st.warning(
                        "'"
                        + pdf_name
                        + "'"
                        + " no se ha podido descargar. Inténtelo de nuevo más tarde."
                    )

            if available_pdfs:
                st.session_state["available_documents"] = True
            else:
                st.warning(
                    "No se han podido descargar los documentos. Inténtelo de nuevo."
                )
                return

            content = extract_text_from_pdf(available_pdfs)
            create_chain_raw(content)
            st.rerun()


def get_example_questions(selected_example):
    if (
        selected_example
        == "Ley 7/2015, de 7 de agosto, de iniciativa legislativa popular y participación ciudadana en el Parlamento de Galicia"
    ):
        return [
            "¿En qué artículo del EEAA (Estatuto de Autonomía de Galicia) se recoge la iniciativa popular?",
            "¿Cómo se regula la iniciativa popular?",
            "¿Existe regulación sobre la iniciativa popular a nivel municipal?",
        ]
    elif (
        selected_example
        == "Modificación del Reglamento Orgánico de Participación Ciudadana del Ayuntamiento de Madrid, de 24 de abril de 2018"
    ):
        return [
            "¿Cuál es el objeto de la modificación?",
            "¿A qué contenido afecta esta modificación?",
        ]
    else:
        return


# --------------------------------------------------------- #


# ---------------------------- RAG -----------------------------
def get_embeddings(type="Nomic"):
    if type == "Nomic":
        from langchain_nomic.embeddings import NomicEmbeddings

        return NomicEmbeddings(model="nomic-embed-text-v1")
    elif type == "OpenAI":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(model="text-embedding-3-small")
    else:
        return None


def get_llm(type="gpt-3.5-turbo"):
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

'''
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
'''

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
    solo reformularla si es necesario y devuelvela."""
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
    embeddings = get_embeddings(EMBEDDINGS_FUNCTION)
    vectorstore = Chroma.from_documents(content, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="similarity_score_threshold", search_kwargs={"k": 3, "score_threshold": QA_THRESHOLD}
    )

    format = itemgetter("docs") | RunnableLambda(format_docs)
    retriever_chain = (RunnableParallel(question=RunnablePassthrough(), docs=retriever).assign(context=format)) #| itemgetter("context")
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

# for some reason the text in the quotes is not being displayed correctly
def convert_to_utf8(text):
    text = text.replace("Ã¡", "á")
    text = text.replace("Ã©", "é")
    text = text.replace("Ã³", "ó")
    text = text.replace("Ãº", "ú")
    text = text.replace("Ã", "í")
    return text

def format_response(data, docs, response):
    # line breaks are done with 2 spaces + \n
    # display answer
    answer = response["answer"]["answer"] + "  \n  \nRespuesta extraída de "

    # display citations
    for i in range(len(response["answer"]["citations"])):
        doc_info = docs[response["answer"]["citations"][i]["source_id"]-1]
        doc = doc_info.metadata["source"]
        # redo the original name - undo the changes made in the download_pdfs function
        doc = doc.split("/")[1].replace("_", " ").replace("-", "/").split(".pdf")[0]
        doc_url = data[data["Norma"] == doc]["URL"].to_list()[0]

        answer += ("["
            + doc
            + "]("
            + doc_url
            + "),"
            + " página "
            + str(doc_info.metadata["page"])
            +":  \n*..."
            + convert_to_utf8(response["answer"]["citations"][i]["quote"])
            + "*  \n\n"
        )

    return answer

def generate_response(data, user_prompt):
    retriever_chain = st.session_state["retriever_chain"]
    qa_chain = st.session_state["qa_chain"]

    logging.info("Calling the retriever chain...")
    retriever_chain_output = retriever_chain.invoke(user_prompt)
    logging.info("Retriever chain output: \n" + str(retriever_chain_output))
    #context = retriever_chain_output | itemgetter("context")
    context = retriever_chain_output['context']
    if context == "No results":
        answer = "Lo siento, no he encontrado nada relacionado con tu consulta."
    else:
        qa_chain_output = qa_chain.invoke({"question":user_prompt, "context":context})
        logging.info("Calling the QA chain...")
        logging.info("QA chain output: \n" + str(qa_chain_output))
        answer = format_response(data, retriever_chain_output['docs'], qa_chain_output)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )
    return answer


# --------------------------------------------------------------
