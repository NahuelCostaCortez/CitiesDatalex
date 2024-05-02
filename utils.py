import time
import pandas as pd
import numpy as np
import streamlit as st
import data
import rag
import logging

logging.basicConfig(level=logging.INFO)


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
    materia_options = ["Cualquiera"]
    if selected_ambito_tematico == "Sostenibilidad Económica":
        materia_options += [
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
        materia_options += [
            "Medio ambiente natural",
            "Medio ambiente urbano",
            "Prevención/gestión riesgos accidentes industriales",
            "Prevención/gestión desastres naturales",
            "Eficiencia en el uso de recursos naturales",
        ]
    elif selected_ambito_tematico == "Cambio Climático":
        materia_options += [
            "Energía",
            "Movilidad urbana",
            "Emisiones de gases de efecto invernadero (GEI)",
            "Efectos peligrosos del cambio climático",
            "Mitigación de los efectos del cambio climático",
            "Adaptación a los efectos del cambio climático",
            "Geoingeniería",
        ]
    elif selected_ambito_tematico == "Sostenibilidad Social":
        materia_options += [
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
        materia_options += [
            "Competencias",
            "Escalas territoriales",
            "Participación ciudadana y de agentes sociales",
            "Planeamiento urbano",
            "Cooperación territorial (Cross border)",
            "Financiación",
            "Regulación",
        ]
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


# ------------------------  BUSCADOR ------------------------ #


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


def search_logic(
    df_data,
    df_tesauro,
    filters,
    search_text,
    selected_ambito_tematico=None,
    selected_ambito_territorial=None,
    selected_escala_normativa=None,
    selected_materia=None,
    selected_submaterias=None,
    selected_comunidad=None,
    selected_municipio=None,
    input_keywords=None,
):
    """
    Perform search logic based on the provided parameters.

    Calls display_results(filtered_rows, selected_ambito_territorial, df_tesauro) to display the filtered results in a form.

    Args:
        df_data (pandas.DataFrame): The data frame to search within.
        df_tesauro (pandas.DataFrame): The thesaurus data frame.
        filters (bool): Indicates whether filters are applied.
        search_text (str): The search text to filter the data frame.
        selected_ambito_tematico (str, optional): The selected thematic scope. Defaults to None.
        selected_ambito_territorial (str, optional): The selected territorial scope. Defaults to None.
        selected_escala_normativa (str, optional): The selected normative scale. Defaults to None.
        selected_materia (str, optional): The selected subject. Defaults to None.
        selected_submaterias (list, optional): The selected sub-subjects. Defaults to None.
        selected_comunidad (str, optional): The selected community. Defaults to None.
        selected_municipio (str, optional): The selected municipality. Defaults to None.
        input_keywords (list, optional): The input keywords for filtering. Defaults to None.

    Returns:
        None
    """

    filtered_rows = None

    # First, filter by search text if any
    if search_text != "" and search_text != None:
        logging.info("Filtering by search text...")
        filtered_rows = rag.filter_by_search(df_data, search_text)
        logging.info(
            "Data filtered by search text, results found: "
            + str(len(filtered_rows))
            + " rows."
        )

    # Then, filter by the selected filters, if any
    if filters:
        logging.info("Filtering by filters...")

        filtered_rows = filter_by_filters(
            filtered_rows if filtered_rows is not None else df_data,
            df_tesauro,
            input_keywords,
            selected_ambito_tematico,
            selected_ambito_territorial,
            selected_escala_normativa,
            selected_materia,
            selected_submaterias,
            selected_comunidad,
            selected_municipio,
        )

    # No results
    if filtered_rows is None or filtered_rows.empty:
        if filters:
            st.write("No hay resultados para los filtros seleccionados")
        else:
            st.write("No hay resultados para la búsqueda realizada")
    else:
        # Save the filtered rows, selected_ambito_territorial and df_tesauro in the session state to use them later if the page is reloaded
        st.session_state["search"] = filtered_rows
        st.session_state["selected_ambito_territorial"] = selected_ambito_territorial
        st.session_state["df_tesauro"] = df_tesauro
        display_results(filtered_rows, selected_ambito_territorial, df_tesauro)


def filter_by_filters(
    df_data,
    df_tesauro,
    input_keywords,
    selected_ambito_tematico,
    selected_ambito_territorial,
    selected_escala_normativa,
    selected_materia,
    selected_submaterias,
    selected_comunidad,
    selected_municipio,
):

    filtered_rows = df_data

    # AMBITO TEMATICO
    if selected_ambito_tematico != "Cualquiera" and selected_ambito_tematico != None:
        # Filter by selected ambito tematico (last columns of the 'Registros' sheet)
        filtered_rows = filtered_rows[
            pd.notna(filtered_rows[selected_ambito_tematico + ".1"])
        ]

    # AMBITO TERRITORIAL
    if (
        selected_ambito_territorial != "Cualquiera"
        and selected_ambito_territorial != None
    ):
        if selected_ambito_territorial == "Autonómico":
            selected_ambito_territorial = "Comunitario"
        filtered_rows = filtered_rows[
            filtered_rows["Ambito Territorial"] == selected_ambito_territorial
        ]

    # ESCALA NORMATIVA
    if selected_escala_normativa != "Cualquiera" and selected_escala_normativa != None:
        selected_escala_normativa = get_selected_escala_normativa(
            selected_escala_normativa
        )
        if selected_escala_normativa == None:
            logging.error("No se ha encontrado la escala normativa seleccionada: ")
        else:
            # Filter by selected escala normativa
            filtered_rows = filtered_rows[
                filtered_rows["Escala Normativa"] == selected_escala_normativa
            ]

    # MATERIA
    if selected_materia != "Cualquiera" and selected_materia != None:
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

    # COMUNIDAD AUTÓNOMA
    if selected_comunidad != "Cualquiera" and selected_comunidad != None:
        print("len filtered rows: ", len(filtered_rows))
        filtered_rows = filtered_rows[filtered_rows["CCAA"] == selected_comunidad]
        print("len filtered rows: ", len(filtered_rows))

    # MUNICIPIO
    if selected_municipio != "Cualquiera" and selected_municipio != None:
        print(filtered_rows["Ciudad"])
        filtered_rows = filtered_rows[filtered_rows["Ciudad"] == selected_municipio]

    # KEYWORDS
    if input_keywords:
        keywords = input_keywords.split(",")
        for i, keyword in enumerate(keywords):
            filter_keyword = filtered_rows[
                filtered_rows["Norma_translated"]
                .str.contains(keyword, case=False)
                .fillna(False)
            ]
            if i == 0:
                filtered_rows = filter_keyword
            # Before concatenating, remove the rows that are already in filtered_rows
            filter_keyword = filter_keyword[
                ~filter_keyword["Norma_translated"].isin(filtered_rows["Norma"])
            ]
            filtered_rows = pd.concat([filtered_rows, filter_keyword])

    return filtered_rows


def display_results(filtered_rows, selected_ambito_territorial, df_tesauro):
    # Display results
    st.markdown("### Resultados")
    st.markdown(str(len(filtered_rows)) + " resultados encontrados.")

    comunities = filtered_rows[filtered_rows["CCAA"].notna()]["CCAA"].unique()
    if len(comunities) > 0:
        # show toggle to display a map with the CCAA
        map = st.toggle("Mostrar en mapa")
        if map:

            # import folium
            # from streamlit_folium import st_folium, folium_static

            st.markdown("Comunidades donde se han encontrado resultados")

            df = get_coordinates(comunities)

            st.map(df, size=20)

            """
            m = folium.Map(
                location=[df.lat.mean(), df.lon.mean()],
                zoom_start=3,
                control_scale=True,
            )

            # Loop through each row in the dataframe
            for i, row in df.iterrows():
                # Setup the content of the popup
                iframe = folium.IFrame(str(row["community"]))

                # Initialise the popup using the iframe
                popup = folium.Popup(iframe, min_width=100, max_width=100)

                # Add each row to the map
                folium.Marker(
                    location=[row["lat"], row["lon"]],
                    popup=popup,
                    c="red",
                ).add_to(m)

            st_data = st_folium(m, width=1000, height=500, zoom=5)
            """

    if selected_ambito_territorial == "CCAA":
        colms = st.columns((2, 1, 1, 1, 1))
        fields = ["Norma", "Ámbito territorial", "CCAA", "Descriptores", "Añadir"]
    elif selected_ambito_territorial == "Local":
        colms = st.columns((2, 1, 1, 1, 1, 1))
        fields = [
            "Norma",
            "Ámbito territorial",
            "CCAA",
            "Municipio",
            "Descriptores",
            "Añadir",
        ]
    else:
        colms = st.columns((2, 1, 1, 1))
        fields = ["Norma", "Ámbito territorial", "Descriptores", "Añadir"]

    # Display headers in bold
    for col, field_name in zip(colms, fields):
        col.markdown("**" + field_name + "**")

    # Form is necessary to keep the checkboxes checked when reloading the page
    checkbox_statusses = []
    with st.form("results", clear_on_submit=False):
        for index, row in filtered_rows.iterrows():

            if selected_ambito_territorial == "Comunitario":
                col1, col2, col3, col4, col5 = st.columns((2, 1, 1, 1, 1))

                # Ambito territorial
                col2.write(row["Ambito Territorial"])

                # CCAA
                if pd.isna(row["CCAA"]):
                    col3.write("No aplica")
                else:
                    col3.write(row["CCAA"])

                # Ambito territorial
                col2.write(row["Ambito Territorial"])
            elif selected_ambito_territorial == "Local":
                col1, col2, col3, col3_1, col4, col5 = st.columns((2, 1, 1, 1, 1, 1))

                # Ambito territorial
                col2.write(row["Ambito Territorial"])

                # CCAA
                if pd.isna(row["CCAA"]):
                    col3.write("No aplica")
                else:
                    col3.write(row["CCAA"])
                # Ciudad
                if pd.isna(row["Ciudad"]):
                    col3_1.write("No aplica")
                else:
                    col3_1.write(row["Ciudad"])
            else:
                col1, col2, col4, col5 = st.columns((2, 1, 1, 1))
                # Ambito territorial
                # &nbsp to add spaces
                if pd.isna(row["CCAA"]):
                    col2.markdown(
                        "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"
                        + row["Ambito Territorial"]
                    )
                else:
                    col2.markdown(
                        # "&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+
                        row["Ambito Territorial"]
                        + ": "
                        + row["CCAA"]
                    )

            # Norma
            col1.markdown(
                "[{0}]({1})".format(row["Norma"], row["URL"]),
                unsafe_allow_html=True,
            )

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
                            )[0]
                            if len(code_row) > 0:
                                code_row = code_row[0]
                                # the name of the label should be to the right
                                code_column += 1
                                while pd.isna(df_tesauro.iloc[code_row, code_column]):
                                    code_column += 1
                                labels.append(df_tesauro.iloc[code_row, code_column])
                            else:
                                logging.error(
                                    "No se ha encontrado el código ",
                                    code,
                                    " en el tesauro.",
                                )

            if len(labels) > 0:
                col4.write(", ".join(labels))

            # Añadir
            checkbox_statusses.append(
                col5.checkbox(
                    "Añadir",
                    key=index,
                    value=st.session_state["checkbox_values"],
                )
            )

        # Add a checkbox to select all the rows
        #checkbox_all = st.checkbox("Añadir todos", key="all")

        # Add a button to submit the form
        submitted = st.form_submit_button(
            "Cargar documento/s",
            help="Se proporcionarán los documentos seleccionados al asistente",
        )
        if st.session_state["documents_loaded"]:
            if st.session_state["erroneous_pdfs"]==[]:
                st.success("Documentos cargados correctamente en el sistema")
            else:
                st.error("No se han podido cargar los siguientes documentos: " + ", ".join(st.session_state["erroneous_pdfs"])+"\n. Puede comprobar los documentos cargados en la pestaña de 'Asistente Virtual'.")
        # when pressing submitted get the values of the checkboxes
        if submitted:
            # all selected
            #if checkbox_all:
            #   checkbox_statusses = [True for i in range(len(filtered_rows))]
                # when pressing the button, the value of all checkboxes is set to True
            #    st.session_state["checkbox_values"] = True
            # "Añadir todos" is not selected
            #else:
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
            available_pdfs = data.download_pdfs(
                filtered_checked_rows["Norma"],
                filtered_checked_rows["URL"].to_list(),
                filtered_checked_rows["ID"].to_list(),
            )

            available_pdfs_names = [pdf[0] for pdf in available_pdfs]
            for pdf_name in filtered_checked_rows["Norma"]:
                if pdf_name not in available_pdfs_names:
                    # escribir mensaje de error en rojo
                    st.warning(
                        "'"
                        + pdf_name
                        + "'"
                        + " no se ha podido descargar. Inténtelo de nuevo más tarde."
                    )

            if not available_pdfs:
                st.warning(
                    "No se han podido descargar los documentos. Inténtelo de nuevo."
                )
                return

            with st.spinner("Cargando documentos en el sistema..."):
                time.sleep(1)
                content, erroneus_pdfs = data.extract_text_from_pdf(available_pdfs_names)
                if content is None or content == []:
                    return
                else:
                    rag.create_chains(content)
                    if erroneus_pdfs != []:
                        available_pdfs = [pdf for pdf in available_pdfs if pdf[0] not in erroneus_pdfs]
                    st.session_state["documents_loaded"] = available_pdfs
                    st.session_state["erroneous_pdfs"] = erroneus_pdfs
                    st.rerun()
# --------------------------------------------------------- #


# ------------------------  Others ------------------------ #


def get_selected_escala_normativa(selected_escala_normativa):
    if selected_escala_normativa == "Directiva Europea":
        return "DIR_UE"
    elif selected_escala_normativa == "Regulación Europea":
        return "REG_UE"
    elif selected_escala_normativa == "Ley Autonómica":
        return "LEY_CCAA"
    elif selected_escala_normativa == "Ley estatal":
        return "LEY_EST"
    elif selected_escala_normativa == "Plan Urbanístico":
        return "PLAN_URB"
    elif selected_escala_normativa == "White paper":
        return "White paper"
    elif selected_escala_normativa == "Comunicación":
        return "Notice"
    elif selected_escala_normativa == "Decisión":
        return "Decisión"
    elif selected_escala_normativa == "Acuerdo institucional":
        return "Institutional Agreement"
    elif selected_escala_normativa == "Documento Nacional":
        return "DOC_NA"
    elif selected_escala_normativa == "Otros":
        return "OTROS"
    return None


def get_municipios(df_data, comunidad):
    municipios_options = ["Cualquiera"]
    if comunidad != "Cualquiera" and comunidad != None:
        # Get municipios of the selected comunidad
        # THESE ARE SELECTED LOOKING AT THE DIFFERENT ELEMENTS IN THE COLUMN "CIUDAD"
        # BUT DOES NOT TAKE INTO ACCOUNT ALL THE MUNICIPIOS OF THE CCAA
        # IF A NEW ROW IS ADDED AND THE MUNICIPIO IS NOT INCLUDED HERE, IT WILL NOT BE SHOWN
        if comunidad == "Andalucía":
            municipios_options += [
                "Almería",
                "Cádiz",
                "Córdoba",
                "Granada",
                "Huelva",
                "Jaén",
                "Málaga",
                "Sevilla",
            ]
        elif comunidad == "Aragón":
            municipios_options += [
                "Teruel",
                "Zaragoza",
            ]
        elif comunidad == "Canarias":
            municipios_options += [
                "Las Palmas",
                "Santa Cruz de Tenerife",
            ]
        elif comunidad == "Cantabria":
            municipios_options += [
                "Santander",
            ]
        elif comunidad == "Castilla y León":
            municipios_options += [
                "Ávila",
                "León",
                "Soria",
                "Zamora",
            ]
        elif comunidad == "Castilla - La Mancha":
            municipios_options += [
                "Ciudad Real",
                "Cuenca",
                "Guadalajara",
                "Toledo",
            ]
        elif comunidad == "Cataluña":
            municipios_options += [
                "Barcelona",
                "Lleida",
                "Tarragona",
                "Gerona",
            ]
        elif comunidad == "Comunitat Valenciana":
            municipios_options += [
                "Alicante/Alacant",
                "Castellón/Castelló",
                "Valencia",
            ]
        elif comunidad == "Extremadura":
            municipios_options += [
                "Badajoz",
                "Cáceres",
            ]
        elif comunidad == "Galicia":
            municipios_options += [
                "La Coruña",
                "Lugo",
                "Ourense",
                "Pontevedra",
            ]
        elif comunidad == "Comunidad de Madrid":
            municipios_options += [
                "Madrid",
            ]
        elif comunidad == "Región de Murcia":
            municipios_options += [
                "Murcia",
            ]
        elif comunidad == "País Vasco":
            municipios_options += ["Bilbao", "Vitoria", "San Sebastían"]
        elif comunidad == "La Rioja":
            municipios_options += [
                "Logroño",
            ]
        elif comunidad == "Illes Balears":
            municipios_options += [
                "Palma de Mallorca",
            ]
        elif comunidad == "Ceuta":
            municipios_options += [
                "Ceuta",
            ]
        elif comunidad == "Melilla":
            municipios_options += [
                "Melilla",
            ]
        return municipios_options
    else:
        # Get all the municipios
        # Obtain the different names of the column "Ciudad"
        municipios = df_data["Ciudad"].unique()
        # Order them in a list in alphabetical order
        municipios_options = np.sort(municipios[1:])
        # Add "Cualquiera" option
        municipios_options = np.insert(municipios_options, 0, "Cualquiera")
        return municipios_options


def get_month(search_text):
    """
    Returns the month name based on the given search text.

    Args:
        search_text (str): The text to search for the month name.

    Returns:
        str: The name of the month if found in the search text, otherwise None.
    """
    months = {
        "enero": "enero",
        "febrero": "febrero",
        "marzo": "marzo",
        "abril": "abril",
        "mayo": "mayo",
        "junio": "junio",
        "julio": "julio",
        "agosto": "agosto",
        "septiembre": "septiembre",
        "octubre": "octubre",
        "noviembre": "noviembre",
        "diciembre": "diciembre",
    }
    for month in months:
        if month in search_text:
            return months[month]
    return None


def get_coordinates(communities):
    coordinates = []
    for community in communities:
        if community == "Andalucía":
            coordinates.append((community, 37.3886, -5.9826))
        elif community == "Aragón":
            coordinates.append((community, 41.6488, -0.8891))
        elif community == "Principado de Asturias":
            coordinates.append((community, 43.3614, -5.8593))
        elif community == "Illes Balears":
            coordinates.append((community, 39.5342, 2.8577))
        elif community == "Canarias":
            coordinates.append((community, 28.2916, -16.6291))
        elif community == "Cantabria":
            coordinates.append((community, 43.1828, -3.9878))
        elif community == "Castilla y León":
            coordinates.append((community, 41.8356, -4.3976))
        elif community == "Castilla - La Mancha":
            coordinates.append((community, 39.8628, -4.0273))
        elif community == "Cataluña":
            coordinates.append((community, 41.5912, 1.5209))
        elif community == "Ceuta":
            coordinates.append((community, 35.8894, -5.3213))
        elif community == "Comunitat Valenciana":
            coordinates.append((community, 39.4840, -0.7533))
        elif community == "Extremadura":
            coordinates.append((community, 38.9197, -6.3430))
        elif community == "Galicia":
            coordinates.append((community, 42.5751, -8.1339))
        elif community == "Comunidad de Madrid":
            coordinates.append((community, 40.4168, -3.7038))
        elif community == "Melilla":
            coordinates.append((community, 35.2923, -2.9381))
        elif community == "Región de Murcia":
            coordinates.append((community, 37.9922, -1.1307))
        elif community == "Comunidad Foral de Navarra":
            coordinates.append((community, 42.6954, -1.6761))
        elif community == "País Vasco":
            coordinates.append((community, 42.9912, -2.6189))
        elif community == "La Rioja":
            coordinates.append((community, 42.2871, -2.5396))
    df = pd.DataFrame(coordinates, columns=["community", "lat", "lon"])
    return df


def get_example_questions(selected_example):
    if (
        selected_example
        == "Ley 7/2015, de 7 de agosto, de iniciativa legislativa popular y participación ciudadana en el Parlamento de Galicia"
    ):
        return [
            "¿En qué artículo del EEAA (Estatuto de Autonomía de Galicia) se recoge la iniciativa popular?",
            "¿Qué instrumento regula la iniciativa popular?",
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


# for some reason the text in the quotes (returned from llm) is not being displayed correctly
def convert_to_utf8(text):
    text = text.replace("Ã¡", "á")
    text = text.replace("Ã©", "é")
    text = text.replace("Ã³", "ó")
    text = text.replace("Ãº", "ú")
    text = text.replace("Ã", "í")
    text = text.replace("\\u00f3", "ó")
    text = text.replace("\\u00e1", "á")
    text = text.replace("\\u00ed", "í")
    text = text.replace("\\u00e9", "é")
    text = text.replace("\\u00fa", "ú")
    text = text.replace("\\u00f1", "ñ")
    return text


# --------------------------------------------------------- #
