import numpy as np
import utils
import streamlit as st
import time


# Page config, this should be at the top of the script
st.set_page_config(
    page_title="Cities DataLex",
    page_icon="https://www.unioviedo.es/urbanred/wp-content/uploads/2023/10/citieslex_logo-1024x464.png",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# -------------------------  DATA ------------------------- #
df_data, df_tesauro = utils.load_data()
# --------------------------------------------------------- #


# ----------------  SESSION STATE VARIABLES --------------- #
# By default, Streamlit reloads the entire page every time a button is pressed.
# Therefore, these are needed to keep the state of the app.

# search button
if "search" not in st.session_state:
    st.session_state["search"] = False

# RAG chain
if "qa_chain" not in st.session_state:
    st.session_state["qa_chain"] = None

# form with filtered documents
if "checkbox_values" not in st.session_state:
    st.session_state["checkbox_values"] = False

# check whether there are documents in which to search for information
if "available_documents" not in st.session_state:
    st.session_state["available_documents"] = False
# --------------------------------------------------------- #


# -------------------------  Page ------------------------- #
st.image(
    "https://www.unioviedo.es/urbanred/wp-content/uploads/2023/10/citieslex_logo-1024x464.png",
    width=200,
)

tab1, tab2 = st.tabs(["Buscador jurídico", "Asistente Virtual"])

with tab1:

    st.title("Buscador Jurídico")
    st.markdown("Normativa relacionada con la sostenibilidad urbana")

    search_text = st.text_input(
        " ",
        key="search_text",
        help="Si no tienes claro qué buscar puedes dejarlo en blanco y directamente usar los filtros",
    )

    filters = st.toggle("Filtros")

    # -----------------------  FILTERS ------------------------ #
    if filters:
        col_ambito, col_materia = st.columns(2)
        col_comunidad, col_municipio = st.columns(2)

        # Ambito temático
        with col_ambito:
            selected_ambito_tematico = st.selectbox(
                "Ámbito Temático",
                options=["Cualquiera"] + utils.ambito_tematico_options(),
                key="ambito_tematico",
            )

        # Materia
        with col_materia:
            selected_materia = st.selectbox(
                "Materia",
                options=["Cualquiera"]
                + utils.get_materia_options(selected_ambito_tematico),
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
            # Obtain the different names of the column "Ciudad"
            municipios = df_data["Ciudad"].unique()
            # Order them in a list in alphabetical order
            municipios_options = np.sort(municipios[1:])
            # Add "Cualquiera" option
            municipios_options = np.insert(municipios_options, 0, "Cualquiera")
            selected_municipio = st.selectbox(
                "Ciudad/Municipio", options=municipios_options, key="municipio"
            )

        # keywords
        input_keywords = st.text_input(
            "Palabras clave (introdúcelas separadas por comas)", key="input_keywords"
        )
        # create 2 buttoms side by side
        col1, col2 = st.columns([10, 1])

        with col2:
            # reset button
            reset_button = st.button("Resetear filtros", on_click=utils.reset_filters)
    # ------------------------------------------------------------- #

    # -------------------------  RESULTS -------------------------- #
    # search button
    search_button = st.button("Buscar")

    if search_button or st.session_state["search"]:
        st.session_state["search"] = True

        if filters:
            with st.spinner("Recuperando información..."):
                time.sleep(1)
                utils.search_logic(
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
                )

        else:
            with st.spinner("Recuperando información..."):
                time.sleep(1)
                utils.search_logic(
                    df_data,
                    df_tesauro,
                    filters,
                    search_text,
                    selected_ambito_tematico=None,
                    selected_materia=None,
                    selected_submaterias=None,
                    selected_comunidad=None,
                    selected_municipio=None,
                    input_keywords=None,
                )


with tab2:

    # -------------------------  CHATBOT -------------------------- #
    st.title("Asistente")
    st.markdown(
        "Puedo ayudarte a encontrar información contenida entre los documentos seleccionados."
    )

    col_upload, col_example = st.columns(2)

    with col_upload:
        st.markdown(
            """... o si lo prefieres, puedes pasarme un documento específico. """
        )
        uploaded_file = st.file_uploader("sube aquí tu pdf", type="pdf")
        # add button to load pdf
        load_button = st.button("Cargar pdf")

    with col_example:
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
        st.session_state["available_documents"] = True
        content = (
            utils.extract_text_from_pdf([selected_example])
            if selected_example
            else utils.extract_text_from_pdf(uploaded_file.name)
        )
        utils.create_chain_raw(content)

    # add divider
    st.markdown("""--------""")

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

        messages = st.container(height=300)

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
                # st.write(user_prompt)
                st.write(user_prompt)

            if (
                st.session_state["qa_chain"] is None
                or st.session_state["available_documents"] == False
            ):
                # with st.chat_message("assistant"):
                with messages.chat_message("assistant"):
                    answer = "Para poder ayudarte primero debes cargar algún documento. Puedes buscar normativa con ayuda de los filtros o subir directamente un documento al sistema."

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

                    st.write(utils.generate_response(df_data, user_prompt))
                    # chat_history = chat_history + [(prompt, response)]
                    # print(chat_history)
    # ------------------------------------------------------------- #
