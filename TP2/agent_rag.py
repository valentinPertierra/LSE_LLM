# iniciar con: streamlit run agent_rag.py


import streamlit as st
import os
from groq import Groq

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.embeddings import HuggingFaceEmbeddings

from langchain.chains import ConversationChain, LLMChain
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory

from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph, END
from langchain import hub
from langchain.prompts import PromptTemplate

import re


from dotenv import load_dotenv





def main():
    """
    Esta función es el punto de entrada principal de la aplicación. Configura el cliente de Groq, 
    la interfaz de Streamlit y maneja la interacción del chat.
    """
    
    load_dotenv()

    # Obtener la clave API de Groq
    GROQ_API_KEY = os.getenv('GROQ_API_KEY')  

    # API key de piencone
    PINECONE_API_KEY=os.getenv("PINECONE_API_KEY")

    #Connect to DB Pinecone
    pc = Pinecone(api_key=PINECONE_API_KEY)
    spec = ServerlessSpec(cloud='aws', region='us-east-1')

    # Cargo un modelo de embeddings compatible
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    

    # Inicializo el agente 
    llm = ChatGroq(
        groq_api_key=GROQ_API_KEY, 
        model_name='llama3-8b-8192'
    )

    # Defino un tamplate para el prompt
    prompt = PromptTemplate(
        input_variables=["context", "question", "individual"],
        template="""
    Eres un asistente para tareas de preguntas y respuestas. Usa los siguientes fragmentos de historia y contexto recuperados para responder la pregunta respecto al individuo.
    Si no sabes la respuesta, di que no lo sabes. Usa un máximo de 200 palabras y mantén la respuesta concisa. 
    ---
    Historia:
    {history}
    ---
    Contexto:
    {context}
    ---
    Individuo:
    {individual}
    ---
    Pregunta: {question}
    Respuesta:
    """
    )

    # Defino una clase para guardar el estado 
    class State(TypedDict):
        question: str
        context: List[Document]
        answer: str
        individual: str
        history: List[str] 


    # Defino una clase agente para hacer la busqueda en la base vectorial segun la persona
    class Agent:
        
        def __init__(self, embedding_model, index="", namespace="espacio"):
            if index=="":
                raise ValueError("No se especifico un índice válido.")
            
            self.index = index
            self.embedding_model = embedding_model

            self.vectorstore = PineconeVectorStore(
                index_name=index,
                embedding=self.embedding_model,
                namespace=namespace,
            )

        def get_context(self,state: State):

            retrieved_docs = self.vectorstore.similarity_search(state["question"],k=2)
            print(retrieved_docs)
            return {"context": retrieved_docs}


    # Defino los nodos para el agente
    def generate(state: State):
        if state["context"]:
            docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        else: 
            docs_content = ""
        # Formateo la historia como un unico string
        history = "\n".join(state["history"])
        
        # Invoco el prompt con contexto e historia previa
        messages = prompt.invoke({
            "question": state["question"],
            "context": docs_content,
            "individual": state["individual"],
            "history": history
        })

        # print(messages)
        response = llm.invoke(messages)
        
        state["history"].append(f"Q: {state['question']} A: {response.content}")
        
        # Ahora ya es posible devolver la respuesta
        return {"answer": response.content}

    # Nodo para limpiar el contexto
    def empty_context(state:State):
        return {"context":[]}

    # Segun sobre a quien se refiere la pregunta se utiliza un agente u otro
    def decide(state: State):
        
        valentin_pattern = r"(Valentin\sPertierra|Valentin|Pertierra)"
        juan_pattern = r"(Juan\sPérez|Juan|Pérez)"
        individual = "" 
        
        if re.search(valentin_pattern, state["question"], re.IGNORECASE):
            individual = "valentin"
        elif re.search(juan_pattern, state["question"], re.IGNORECASE):
            individual = "juan"
        return {"individual":individual}

    # Funcion para determinar cuál es el próximo nodo
    def decision_read_state(state:State):
        """Obtiene el individuo desde el state y lo retorna para decidir por qué nodo continuar."""
        indiv = state["individual"]
        if indiv=="":
            print("La pregunta no habla de ningun individuo.")
            return "no_individual"
        print("La pregunta habla sobre el individuo:",indiv)
        return indiv


    

    # Instancio agentes
    vagent = Agent(embedding_model,"vagent")
    jagent = Agent(embedding_model,"jagent")

    # Armo el grafo de nodos
    graph_builder = StateGraph(State)
    graph_builder.add_node("decision",decide)
    graph_builder.add_node("valentin_agent",vagent.get_context)
    graph_builder.add_node("juan_agent",jagent.get_context)
    graph_builder.add_node("generate",generate)
    graph_builder.add_conditional_edges(
        "decision",
        decision_read_state,
        {"valentin": "valentin_agent","juan": "juan_agent","no_individual":"valentin_agent"}
        )
    graph_builder.add_edge("valentin_agent","generate")
    graph_builder.add_edge("juan_agent","generate")
    graph_builder.set_entry_point("decision")
    graph = graph_builder.compile()



    # Aplicacion de Streamlit   
    st.set_page_config(
        page_title="Chatbot Agent RAG",
        page_icon="🤖",
        #layout="wide"
    )

    # El título y mensaje de bienvenida de la aplicación Streamlit
    st.title("Chatbot Agent RAG")
    st.write("¡Hola! Este es un chatbot para interactuar con los CV de Valentin y Juan. Pregunta lo que quieras!")


    user_question = st.text_input("Haz una pregunta:")

    # Variable de estado de la sesión
    if "historial_chat" not in st.session_state:
        st.session_state.conversation_history = {
            "question": "",
            "context": [],
            "answer": "",
            "history": []  
        }

   

    # Si el usuario ha hecho una pregunta,
    if user_question:

        st.session_state.conversation_history["question"] = user_question
        print(st.session_state.conversation_history)
        response = graph.invoke(st.session_state.conversation_history)
        st.write("Chatbot:", response["answer"])






if __name__ == "__main__":
    main()
