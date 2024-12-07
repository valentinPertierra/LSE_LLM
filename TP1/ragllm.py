# iniciar con: streamlit run chatbot_gestionada.py


import streamlit as st
from groq import Groq
import random

from langchain.chains import ConversationChain, LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import os

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings


from dotenv import load_dotenv





def main():
    """
    Esta función es el punto de entrada principal de la aplicación. Configura el cliente de Groq, 
    la interfaz de Streamlit y maneja la interacción del chat.
    """
    
    load_dotenv()

    # Obtener la clave API de Groq
    groq_api_key = os.getenv('GROQ_API_KEY')  

    # API key de piencone
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

    #Connect to DB Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud='aws', region='us-east-1')

    # Cargo un modelo de embeddings compatible
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    vectorstore = PineconeVectorStore(
        index_name='raglse',
        embedding=embedding_model,
        namespace='espacio',
    )

    # Aplicacion de Streamlit   
    st.set_page_config(
        page_title="Chatbot RAG",
        page_icon="🤖",
        layout="wide"
    )

    # El título y mensaje de bienvenida de la aplicación Streamlit
    st.title("Chatbot RAG")
    st.write("¡Hola! Este es un chatbot para interactuar con archivos PDF. Carga un archivo y pregunta lo que quieras!")

    # Agregar opciones de personalización en la barra lateral
    st.sidebar.title('Personalización')
       
    #pdf_file = st.sidebar.file_uploader("Sube un archivo PDF", type=["pdf"])
    #if pdf_file is not None:
    #    save_pdf(pdf_file, "pdfs")

    #if st.sidebar.button("Agregar a base vectorial"):
    #    load_vectordb(pdf_file)

    model = st.sidebar.selectbox(
        'Elige un modelo',
        ['llama3-8b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it']
    )
    conversational_memory_length = st.sidebar.slider('Longitud de la memoria conversacional:', 1, 10, value = 5)

    memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="historial_chat", return_messages=True)

    user_question = st.text_input("Haz una pregunta:")

    # Variable de estado de la sesión
    if 'historial_chat' not in st.session_state:
        st.session_state.historial_chat=[]
    else:
        for message in st.session_state.historial_chat:
            memory.save_context(
                {'input': message['humano']},
                {'output': message['IA']}
            )


    # Inicializar el objeto de chat Groq con Langchain
    groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
    )


    # Si el usuario ha hecho una pregunta,
    if user_question:

        # Busco contenido similar en la base de datos
        context = vectorstore.similarity_search(user_question, k=3)
        #print("context",context)
        system_prompt = f'Eres un asistente para tareas de preguntas y respuestas. \
        Usa los siguientes fragmentos de contexto recuperados para responder la pregunta. \
        Si no sabes la respuesta, di que no lo sabes. Usa un máximo de tres oraciones y mantén la respuesta concisa. \n\n \
        {context}'

        # Construir una plantilla de mensaje de chat utilizando varios componentes
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(
                    content=system_prompt
                ),  # Este es el mensaje del sistema persistente que siempre se incluye al inicio del chat.

                MessagesPlaceholder(
                    variable_name="historial_chat"
                ),  # Este marcador de posición será reemplazado por el historial de chat real durante la conversación. Ayuda a mantener el contexto.

                HumanMessagePromptTemplate.from_template(
                    "{human_input}"
                ),  # Esta plantilla es donde se inyectará la entrada actual del usuario en el mensaje.
            ]
        )

        # Crear una cadena de conversación utilizando el LLM (Modelo de Lenguaje) de LangChain
        conversation = LLMChain(
            llm=groq_chat,  # El objeto de chat Groq LangChain inicializado anteriormente.
            prompt=prompt,  # La plantilla de mensaje construida.
            verbose=True,   # Habilita la salida detallada, lo cual puede ser útil para depurar.
            memory=memory,  # El objeto de memoria conversacional que almacena y gestiona el historial de la conversación.
        )
        
        # La respuesta del chatbot se genera enviando el mensaje completo a la API de Groq.
        try:
            response = conversation.predict(human_input=user_question)
            message = {'humano': user_question, 'IA': response}
            st.session_state.historial_chat.append(message)

        except Exception as e:
            response = f"Error! {e}"

        st.write("Chatbot:", response)


def read_pdf(pdf_file):

    pdf_reader = PyPDF2.PdfReader(pdf_file)
    texto = ""
    for page in pdf_reader.pages:
        texto += page.extract_text()
    
    return texto


def save_pdf(pdf_file, output_dir="pdfs"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Borro todo lo que habia antes, por ahora solo funciona con un solo documento  
    for archivo in os.listdir(output_dir):
        archivo_path = os.path.join(output_dir, archivo)
        if os.path.isfile(archivo_path):
            os.remove(archivo_path)
                
    file_path = os.path.join(output_dir, pdf_file.name)
    with open(file_path, "wb") as f:
        f.write(pdf_file.getbuffer())
    
    return file_path


# Genero los chunks
def chunkData(docs, chunk_size=100, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(docs)
    return chunks


# Cargo el archivo pdf a la base de datos vectorial
def load_vectordb(filePath):
    
    print("aca deberia hacer la logica para cargar el documento a la base de datos vectorial")
    '''
    # Cargo el documento
    loader = PyPDFLoader(filePath)
    docs = loader.load()

    # Genero los chunks 
    chunks = chunkData(docs, chunk_size=200, chunk_overlap=100)
    '''



if __name__ == "__main__":
    main()
