import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import streamlit as st
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# read all pdf files and return text


def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# split text into chunks


def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000, chunk_overlap=1000)
    chunks = splitter.split_text(text)
    return chunks  # list of strings

# get embeddings for each chunk


def get_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")


def get_conversational_chain():
    prompt_template = """
Tu es un assistant spécialisé dans la loi de finance marocaine, conçu exclusivement pour répondre aux questions basées sur le document PDF officiel de la loi de finance fourni.

RÈGLES STRICTES À RESPECTER :
1. Réponds de façon aussi détaillée que possible en utilisant UNIQUEMENT les informations contenues dans le contexte fourni.
2. Si une information n'est pas présente dans le contexte, indique clairement: "Je ne trouve pas d'information concernant votre question dans la loi de finance à ma disposition. Puis-je vous aider sur un autre aspect de la législation financière marocaine?"
3. Ne fais JAMAIS de suppositions ou n'utilise pas de connaissances externes au contexte.
4. Cite SYSTÉMATIQUEMENT les références précises pour chaque réponse (numéro d'article, chapitre, section, etc.)
5. Format de citation obligatoire: "Selon l'article [X] de la loi de finance, [citation exacte]"

STRUCTURE DE TES RÉPONSES :
1. Commence par une réponse directe à la question
2. Cite la référence légale exacte (numéro de loi, article, paragraphe)
3. Explique en détail toutes les dispositions pertinentes avec leurs références respectives
4. Assure-toi d'inclure tous les détails disponibles dans le contexte

RÉPONSE AUX QUESTIONS HORS CONTEXTE :
Si la question ne concerne pas la loi de finance marocaine ou demande des informations clairement hors sujet, réponds poliment: "Cette question ne semble pas porter sur la loi de finance marocaine. Je suis spécialisé dans ce domaine précis. Puis-je vous aider avec une question concernant la fiscalité ou les dispositions financières prévues par la loi de finance?"

RÉPONSE AUX QUESTIONS SANS CORRESPONDANCE :
Si après recherche approfondie, tu ne trouves aucune information correspondante dans le contexte, réponds: "Après analyse complète des documents disponibles, je ne trouve pas de disposition spécifique concernant votre question dans la loi de finance. Souhaitez-vous des informations sur un autre aspect de la législation financière?"

Contexte:
{context}

Question:
{question}
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash",
                                   client=genai,
                                   temperature=0.3,
                                   )
    prompt = PromptTemplate(template=prompt_template,
                            input_variables=["context", "question"])
    chain = load_qa_chain(llm=model, chain_type="stuff", prompt=prompt)
    return chain


def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "upload document and ask me a question"}]


def user_input(user_question):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")  # type: ignore

    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True) 
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()

    response = chain(
        {"input_documents": docs, "question": user_question}, return_only_outputs=True, )

    print(response)
    return response


def main():
    st.set_page_config(
        page_title="Assistance sur la Loi de Finances ",
        page_icon="🏦"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Téléchargez votre document et cliquez sur le bouton Soumettre & Traiter", accept_multiple_files=True)
        if st.button("Soumettre et Traiter"):
            with st.spinner("En cours de traitement..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Votre Assistant IA pour la Loi de Finances 📊")
    st.write("Bienvenue dans le chat !")
    st.sidebar.button('Réinitialiser la conversation', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "Téléchargez un document et posez-moi une question"}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

    # Display chat messages and bot response
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = user_input(prompt)
                placeholder = st.empty()
                full_response = ''
                for item in response['output_text']:
                    full_response += item
                    placeholder.markdown(full_response)
                placeholder.markdown(full_response)
        if response is not None:
            message = {"role": "assistant", "content": full_response}
            st.session_state.messages.append(message)


if __name__ == "__main__":
    main()
