CITIES DATALEX® es un software cuyo propósito es mejorar el acceso a la normativa jurídica resultante de la aplicación en las actuaciones en el medio urbano y, en general, en los procesos de desarrollo urbano y territorial sostenible. CITIES DATALEX® surge como iniciativa conjunta entre las cátedras Concepción Arenal de Agenda 2030 y TotalEnergies de Analítica de Datos e IA con el objetivo de ofrecer a las empresas, entidades financieras y organizaciones privadas que dan soporte a las acciones de las Administraciones Públicas información jurídica clara y segura para desarrollar actividades que tengan repercusión urbanística y territorial.

Este repositorio contiene la funcionalidad base de CITIES DATALEX® como sistema RAG (Retrieval Augmented Generation). El propósito de hacer el software de código abierto es poder extender la aplicabilidad del sistema a cualquier otro ámbito de índole similar donde el objetivo sea búsqueda de información sobre una base de datos de documentos.

La manera más rápida de instalar el software es hacerlo mediante un entorno virtual en python con conda:

```
conda create -n cities-datalex python=3.10
conda activate cities-datalex
pip install -r requirements.txt
```

>[!IMPORTANT]
>- Por defecto el requirements.txt incluye el paquete pysqlite3-binary. Está incluido porque es necesario en la máquina donde está alojado el sistema. Sin embargo, en otras máquinas no lo es, por lo que si te da problemas, lo mejor es simplemente comentarlo del requirements.txt para omitir su instalación.
>- Algunos paquetes hacen uso de comandos de unix, por tanto, es probable que no funcione en windows. Recomiendo utilizar una máquina linux o en su defecto WSL en windows.
>- Se da la opción de utilizar embeddings de OpenAI y de NOMIC, para ello se necesita crear una carpeta '.streamlit' y dentro de ella crear un archivo 'secrets.toml' en el que se especifiquen las api keys. Ejemplo:
> ```OPENAI_API_KEY = "sk-..."```, sustituyendo los puntos suspensivos por la api key en cuestión.