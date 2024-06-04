import numpy as np
import data
import utils
import rag
import streamlit as st
import time
import logging
import os

logging.basicConfig(level=logging.INFO)

os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
os.environ["NOMIC_API_KEY"] = st.secrets["NOMIC_API_KEY"]


# Page config, this should be at the top of the script
st.set_page_config(
    page_title="Cities DataLex",
    page_icon="https://www.unioviedo.es/urbanred/wp-content/uploads/2023/10/citieslex_logo-1024x464.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------  DATA ------------------------- #
df_data, df_tesauro = data.load_data()
# create vector store for the norms if it does not exist
rag.create_vector_store(df_data)
# --------------------------------------------------------- #


# ----------------  SESSION STATE VARIABLES --------------- #
# By default, Streamlit reloads the entire page every time a button is pressed.
# Therefore, these are needed to keep the state of the app.

# search button - this is used to store the search results
if "search" not in st.session_state:
    st.session_state["search"] = None
    retrieved_search = None
    retrieved_selected_ambito_territorial = None
    retrieved_df_tesauro = None

# LLM
if "llm" not in st.session_state:
    st.session_state["llm"] = rag.get_llm(rag.LLM_MODEL)

# RAG chain
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

# form with filtered documents
if "checkbox_values" not in st.session_state:
    st.session_state["checkbox_values"] = False

# check whether documents have been loaded
if "documents_loaded" not in st.session_state:
    st.session_state["documents_loaded"] = False

# check whether a document could not be loaded
if "erroneous_pdfs" not in st.session_state:
    st.session_state["erroneous_pdfs"] = []
# --------------------------------------------------------- #


# -------------------------  Page ------------------------- #
#st.image(
#    "https://www.unioviedo.es/urbanred/wp-content/uploads/2023/10/citieslex_logo-1024x464.png",
#    width=200,
#)

st.image(
    "./logos/urban_red.png",
    width=350,
)

tab_buscador, tab_asistente, tab_info = st.tabs(
    ["Buscador Jurídico", "Asistente", "Información"]
)

with tab_buscador:

    st.title("Buscador Jurídico")
    st.markdown("Busca normativa relacionada con la sostenibilidad urbana")

    search_text = st.text_input(
        " ",
        key="search_text",
        help="Si no tienes claro qué buscar puedes dejarlo en blanco y directamente usar los filtros",
    )

    filters = st.toggle("Filtros")

    # -----------------------  FILTERS ------------------------ #
    if filters:
        # Initialize variables
        selected_ambito_tematico = "Cualquiera"
        selected_ambito_territorial = "Cualquiera"
        selected_materia = "Cualquiera"
        selected_comunidad = "Cualquiera"
        selected_municipio = "Cualquiera"

        col_ambito_tematico, col_ambito_territorial, col_escala_normativa = st.columns(
            3
        )
        col_comunidad, col_municipio = st.columns(2)
        col_materia, col_keywords = st.columns(2)

        # ----------------------- First row ----------------------- #
        # Ambito temático
        with col_ambito_tematico:
            selected_ambito_tematico = st.selectbox(
                "Ámbito Temático",
                options=["Cualquiera"] + utils.ambito_tematico_options(),
                key="ambito_tematico",
            )

        # Ambito territorial
        with col_ambito_territorial:
            selected_ambito_territorial = st.selectbox(
                "Ámbito Territorial",
                options=["Cualquiera", "Europeo", "Estatal", "Autonómico", "Local"],
                key="ambito_territorial",
            )

        # Escala normativa
        with col_escala_normativa:
            selected_escala_normativa = st.selectbox(
                "Escala Normativa",
                options=[
                    "Cualquiera",
                    "Directiva Europea",
                    "Regulación Europea",
                    "Ley Autonómica",
                    "Ley estatal",
                    "Plan Urbanístico",
                    "White paper",
                    "Comunicación",
                    "Decisión",
                    "Acuerdo institucional",
                    "Documento Nacional",
                    "Otros",
                ],
                key="escala_normativa",
            )
        # ---------------------------------------------------------- #

        # ----------------------- Second row ----------------------- #
        if (
            selected_ambito_territorial == "Estatal"
            or selected_ambito_territorial == "Autonómico"
            or selected_ambito_territorial == "Local"
        ):

            # Comunidad Autónoma
            with col_comunidad:
                comunidades_autonomas = df_data["CCAA"].unique()
                comunidades_autonomas_options = np.sort(comunidades_autonomas[1:])

                comunidad_autonoma_options = np.insert(
                    comunidades_autonomas_options, 0, "Cualquiera"
                )
                selected_comunidad = st.selectbox(
                    "Comunidad Autónoma",
                    options=comunidad_autonoma_options,
                    key="comunidad",
                )

            # Municipio
            with col_municipio:
                selected_municipio = st.selectbox(
                    "Municipio",
                    options=utils.get_municipios(df_data, selected_comunidad),
                    key="municipio",
                )
        # ---------------------------------------------------------- #

        # ----------------------- Third row ----------------------- #
        # Materia
        with col_materia:
            selected_materia = st.selectbox(
                "Materia",
                options=utils.get_materia_options(selected_ambito_tematico),
                key="materia",
            )

            selected_submaterias = []

            if selected_materia != "Cualquiera":
                # Check if there are submaterias
                # DE MOMENTO SOLO SE MUESTRAN LAS SUBMATERIAS DE NIVEL 1
                submaterias = utils.get_submateria(
                    df_tesauro, selected_ambito_tematico, selected_materia, nivel=1
                )
                if submaterias != []:
                    toggle = st.toggle("Ver descriptores")
                    if toggle:
                        selected_submaterias = st.multiselect(
                            "Submateria", options=submaterias, placeholder=""
                        )
        with col_keywords:

            # keywords
            input_keywords = st.text_input(
                "Palabras clave (introdúcelas separadas por comas)",
                key="input_keywords",
            )
        # ---------------------------------------------------------- #

        # create 2 columns layout for putting the reset button to the right
        _, col_reset = st.columns([10, 1])

        with col_reset:
            # reset button
            reset_button = st.button("Resetear filtros", on_click=utils.reset_filters)
    # ------------------------------------------------------------- #

    # -------------------------  RESULTS -------------------------- #
    # search button
    search_button = st.button("Buscar")

    if search_button:
        # if documents are already loaded, reset the session state
        st.session_state["documents_loaded"] = False

        if filters and (
            selected_ambito_tematico != "Cualquiera"
            or selected_ambito_territorial != "Cualquiera"
            or selected_escala_normativa != "Cualquiera"
            or selected_materia != "Cualquiera"
            or selected_comunidad != "Cualquiera"
            or selected_municipio != "Cualquiera"
            or input_keywords != ""
        ):
            with st.spinner("Recuperando información..."):
                time.sleep(0.5)
                utils.search_logic(
                    df_data,
                    df_tesauro,
                    filters,
                    search_text,
                    selected_ambito_tematico,
                    selected_ambito_territorial,
                    selected_escala_normativa,
                    selected_materia,
                    selected_submaterias,
                    selected_comunidad,
                    selected_municipio,
                    input_keywords,
                )

        else:
            if search_text:
                with st.spinner("Recuperando información..."):
                    time.sleep(0.5)
                    utils.search_logic(
                        df_data,
                        df_tesauro,
                        filters,
                        search_text,
                    )
            else:
                st.warning("Por favor, introduce algún término de búsqueda o filtro")

    # Check if there are previous results and display them, otherwise the page will be reloaded and no results will be shown
    else:
        retrieved_search = st.session_state["search"]
        if retrieved_search is not None:
            logging.info("Retrieving last search")
            retrieved_selected_ambito_territorial = st.session_state[
                "selected_ambito_territorial"
            ]
            retrieved_df_tesauro = st.session_state["df_tesauro"]
            utils.display_results(
                retrieved_search,
                retrieved_selected_ambito_territorial,
                retrieved_df_tesauro,
            )


with tab_asistente:

    # -------------------------  CHATBOT -------------------------- #
    st.title("Asistente")
    st.markdown(
        "Puedes preguntarle al asistente información relacionada con los documentos seleccionados en la anterior pestaña *'Buscador Jurídico'*"
    )

    st.markdown("""--------""")

    col_upload, col_chat = st.columns(spec=[0.4, 0.6])

    with col_upload:
        st.markdown("""Si lo prefieres, puedes pasarme un documento específico. """)
        uploaded_file = st.file_uploader("sube aquí tu pdf", type="pdf")
        # add button to load pdf
        load_button = st.button("Cargar pdf")

        st.markdown("""... o elegir entre los documentos de ejemplo.""")
        # with col_example:
        selected_example = st.selectbox(
            "Documentos de ejemplo",
            options=[
                # "Decreto 258/2011, de 26 de octubre, por el que se regula la composición, competencias y funcionamiento de la Comisión de Urbanismo y Ordenación del Territorio del Principado de Asturias",
                # "Real Decreto 363/1995, de 10 de marzo, por el que se aprueba el Reglamento sobre notificación de sustancias nuevas y clasificación, envasado y etiquetado de sustancias peligrosas",
                "Ley 7/2015, de 7 de agosto, de iniciativa legislativa popular y participación ciudadana en el Parlamento de Galicia",
                "Modificación del Reglamento Orgánico de Participación Ciudadana del Ayuntamiento de Madrid, de 24 de abril de 2018",
                # "Código de Urbanismo del Principado de Asturias",
            ],
        )

        if selected_example:
            selected_question = st.selectbox(
                "Preguntas de ejemplo",
                options=utils.get_example_questions(selected_example),
            )
            st.code(selected_question, language=None)
            load_button = st.button("Cargar documento")

    if uploaded_file or load_button:
        content, erroneous_pdf = (
            data.extract_text_from_pdf([selected_example])
            if selected_example
            else data.extract_text_from_pdf(uploaded_file.name)
        )
        if erroneous_pdf:
            st.error(f"No se puede procesar el documento '{erroneous_pdf[0]}'")
        rag.create_chains(content)
        st.session_state["documents_loaded"] = [
            (
                selected_example if selected_example else uploaded_file.name,
                "",
            )
        ]

    st.markdown("""--------""")

    with col_chat:

        available_pdfs = st.session_state["documents_loaded"]
        if available_pdfs != False:
            # name[0] for the names, name[1] for the urls
            st.markdown(
                "📄 **Documentos cargados**  \n"
                + "  \n".join(
                    [
                        (
                            "- [" + str(name[0]) + "](" + name[1] + ")"
                            if name[1] != ""
                            else "- " + str(name[0])
                        )
                        for name in available_pdfs
                    ]
                )
            )
        else:
            st.write("⚠️ :red[Actualmente no hay ningún documento cargado]")

        reset_conversation = st.button("Borrar conversación")
        if reset_conversation:
            st.session_state.messages = [
                {"role": "assistant", "content": "¿Cómo puedo ayudarte?"}
            ]

        chat_history = []

        # Store LLM generated responses
        if "messages" not in st.session_state.keys():
            st.session_state.messages = [
                {"role": "assistant", "content": "¿Cómo puedo ayudarte?"}
            ]
        else:
            chat_history = st.session_state.messages

        with st.container():

            messages = st.container(height=500)

            # Display chat messages if any
            for message in st.session_state.messages:
                # with st.chat_message(message["role"]):
                with messages.chat_message(message["role"]):
                    st.write(message["content"])

            # User-provided prompt
            user_prompt = st.chat_input(placeholder="Escriba aquí su consulta")

            if user_prompt:
                # Add user message to chat history and display it
                # with st.chat_message("user"):
                with messages.chat_message("user"):
                    st.session_state.messages.append(
                        {"role": "user", "content": user_prompt}
                    )
                    st.write(user_prompt)

                if (
                    st.session_state["qa_chain"] is None
                    or st.session_state["documents_loaded"] == False
                ):
                    with messages.chat_message("assistant"):
                        answer = "Para poder ayudarte primero debes cargar algún documento. Puedes buscar normativa con ayuda de los filtros en la pestaña 'Buscador Jurídico', elegir alguno de los documentos de ejemplo o subir directamente un documento al sistema."

                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": answer,
                            }
                        )
                        st.write(answer)

                else:
                    # Add assistant message to chat history and display it
                    with messages.chat_message("assistant"):
                        # response = utils.generate_response(
                        #    st.session_state["qa_chain"], user_prompt, chat_history
                        # )
                        with st.spinner("Analizando información..."):
                            time.sleep(0.5)
                            response = rag.generate_response(df_data, user_prompt)
                            st.write(response)
    # ------------------------------------------------------------- #

with tab_info:
    st.write(
        "CITIES DATALEX® es un software cuyo propósito es mejorar el acceso a la normativa jurídica resultante de la aplicación en las actuaciones en el medio urbano y, en general, en los procesos de desarrollo urbano y territorial sostenible. CITIES DATALEX® surge como iniciativa conjunta entre las cátedras Concepción Arenal de Agenda 2030 y TotalEnergies de Analítica de Datos e IA con el objetivo de ofrecer a las empresas, entidades financieras y organizaciones privadas que dan soporte a las acciones de las Administraciones Públicas información jurídica clara y segura para desarrollar actividades que tengan repercusión urbanística y territorial."
    )
    st.write(
        "Actualmente el software se encuentra en fase de desarrollo. Esta versión cuenta con las siguientes funcionalidades:"
    )
    st.markdown(
        "- **Buscador Jurídico**: permite buscar normativa relacionada con la sostenibilidad urbana. La barra superior permite realizar búsquedas semánticas, que pueden ser complementadas junto a los filtros para refinar la búsqueda. **No olvides darle al botón 'Buscar' cada vez que cambies algún filtro.** Sobre cada uno de los resultados de la búsqueda aparecerá un botón de 'Añadir' y al final de la página otro de 'Cargar documentos'. El primero permite añadir documentos a la lista de documentos seleccionados y el segundo permite cargar los documentos seleccionados para que el asistente pueda buscar información en ellos."
    )
    st.write(
        "- **Asistente**: permite consultar información contenida entre los documentos seleccionados. También ofrece la alternativa de cargar documentos en formato PDF o seleccionar documentos de ejemplo."
    )
    st.markdown("""--------""")

column1, column2 = st.columns([1, 0.3])


with column2:
    column2_1, column2_2 = st.columns([1, 1])
    with column2_1:
        st.image("./logos/catedra_concepcion_arenal.png", width=150)
    with column2_2:

        st.image(
            "./logos/totalenergies3-1.png",
            width=100,
        )
