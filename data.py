import os
import re
import streamlit as st
import pandas as pd
import requests
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
import logging

logging.basicConfig(level=logging.INFO)

FOLDER_PATH = "documents"


@st.cache_data  # for not loading the data every time the page is refreshed
def load_data():
    """
    Load data from csv files.

    Returns:
        df_data (pandas.DataFrame): The data from the 'Registros' sheet.
        df_tesauro (pandas.DataFrame): The data from the 'Tesauro' sheet.
    """
    logging.info("Loading data...")

    # Sheet with norms
    # df_data = pd.read_excel("./data/data.csv")
    df_data = pd.read_excel("./data/data.xlsx", engine="openpyxl", sheet_name="data")

    # Some rows have erroneous URLs
    with open("./data/erroneous_urls.txt", "r") as file:
        erroneous_urls = file.readlines()

    # We will remove these rows from the data
    indexes_to_remove = []
    for url in erroneous_urls:
        url = url.strip("\n")
        # Get the indexes of the rows with the erroneous URL
        indexes = df_data[df_data["URL"] == url].index.to_list()
        indexes_to_remove.extend(indexes)

    # Once we have all the indexes, we remove the rows
    df_data = df_data.drop(indexes_to_remove)

    # Sheet with tesauro
    df_tesauro = pd.read_excel(
        "./data/Datos_DL_2022_Final.xlsm", engine="openpyxl", sheet_name="Tesauro"
    )

    logging.info("Data loaded.")

    return df_data, df_tesauro


def download_pdfs(names, urls):
    """
    Download PDF files from the given URLs and save them with corresponding names.

    Args:
        names (list): List of names to use for renaming downloaded files.
        urls (list): List of URLs of the PDF files to download.

    Returns:
        pdf_names (list): List of tuples with names and urls of the downloaded PDF files.
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
            logging.info(f"PDF for '{name}' already exists")
            pdf_names.append((name, url))
            continue
        # Send request to download PDF file
        else:
            logging.info(f"Downloading PDF for '{name}' from URL: {url}")
            try:
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
                    pdf_names.append((name, url))
                else:
                    logging.error(
                        "Error downloading the contents of ",
                        name,
                        ". Try again later.",
                    )
            except Exception as e:
                logging.error(
                    "Error downloading the contents of ", name, ". Traceback: ", e
                )

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


def clean_text(text):
    pattern = r"[^\n0-9a-zA-Z,. áéíóúÁÉÍÓÚ]"  # Matches any character that is not a number, comma, or space
    cleaned_text = re.sub(pattern, "", text)
    # replace \n with space
    cleaned_text = cleaned_text.replace("\n", " ")
    return cleaned_text


def extract_text_from_pdf(pdf_names):

    logging.info("Extracting text from PDFs...")

    content = None

    # user has uploaded a single pdf
    if type(pdf_names) != list:
        loader = PyPDFLoader(pdf_names)
        try:
            content = loader.load_and_split()
        except Exception as e:
            logging.error(f"Error extracting text from PDF: {e}")

    # user has selected/downloaded multiple pdfs
    else:
        content = []
        # Load documents
        for index, pdf_name in enumerate(pdf_names):
            pdf_name = pdf_name.replace(" ", "_")
            pdf_name = pdf_name.replace("/", "-")
            pdf_name = os.path.join(FOLDER_PATH, f"{pdf_name}.pdf")
            loader = PyPDFLoader(pdf_name)
            try:
                pages = loader.load_and_split()
            except Exception as e:
                logging.error(f"Error extracting text from PDF: {e}")
            if index == 0:
                content = pages
            else:
                [content.append(page) for page in pages]

        # Empty document directory
        # for file in os.listdir(FOLDER_PATH):
        #    os.remove(os.path.join(FOLDER_PATH, file))

    # clean the text in content before splitting
    for document in content:
        document.page_content = clean_text(document.page_content)

    text_splitter = CharacterTextSplitter(
        chunk_size=500, chunk_overlap=0, separator="\n"
    )
    content = text_splitter.split_documents(content)

    logging.info("Text extracted from PDFs.")

    return content
