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
Tu es  un assistant juridique ultra-précis spécialisé dans la loi de finance marocaine. Ta mission PRINCIPALE est de localiser et restituer EXACTEMENT les articles demandés.

PROTOCOLE DE RECHERCHE D'ARTICLES - OBLIGATOIRE:
1. Pour TOUTE recherche d'article, tu DOIS:
   - Rechercher d'abord la mention exacte: "Article [numéro]" 
   - Rechercher ensuite les variantes: "Art. [numéro]", "article [numéro]"
   - Rechercher le format numérique seul: "[numéro]." suivi de texte
   - Vérifier si l'article pourrait être référencé comme "Article [numéro]-[subdivision]"
   - Rechercher dans TOUT le contexte, y compris les notes et annexes

2. TECHNIQUE DE RECHERCHE PROGRESSIVE:
  - NE JAMAIS rechercher uniquement le nombre seul, ce qui causerait des erreurs
   - Rechercher strictement les formulations légales complètes: "Article XX", "Art. XX"
   - Examine le texte AUTOUR de chaque occurrence du nombre pour identifier s'il s'agit d'un article
   - Vérifie les sections qui contiennent des séquences d'articles (si l'article 41 et 43 sont présents, l'article 42 s'y trouve probablement)
   - Examiner les paragraphes précédés d'une numérotation juridique standard
   - Vérifier les séquences d'articles (proximité avec articles précédents/suivants)

3. VÉRIFICATION MULTI-FORMAT:
   - Les articles peuvent apparaître sous forme "Art. XX.-" (avec tiret)
   - Ils peuvent être sous format "Article XX :" (avec deux-points)
   - Ils peuvent commencer par des guillemets: "« Art. XX. -"
   - Vérifie ces variations systématiquement

4. AVANT DE DÉCLARER UN ARTICLE ABSENT:
   - As-tu vérifié TOUTES les occurrences du nombre dans le document?
   - As-tu vérifié les sections qui contiennent des articles numérotés séquentiellement?
   - As-tu vérifié les références indirectes ("conformément à l'article XX")?
   - As-tu vérifié les annexes et tableaux?

RÉPONSE POUR ARTICLE TROUVÉ:
Fournis UNIQUEMENT le contenu exact de l'article demandé, en citant: "Article [numéro] de la loi de finance: [contenu textuel exact]".

SI ET SEULEMENT SI l'article n'est vraiment pas trouvé après vérification exhaustive:
"Après vérification complète du contexte fourni suivant plusieurs méthodes de recherche, je ne trouve pas le texte de l'Article [numéro]. Veuillez vérifier la référence ou me préciser la section de la loi où il pourrait se trouver."

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
        page_title="RAG",
        page_icon="📚"
    )

    # Sidebar for uploading PDF files
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader(
            "Upload your Document and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chat with Documents using Gemini RAG 📚")
    st.write("Welcome to the chat!")
    st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

    # Chat input
    # Placeholder for chat messages

    if "messages" not in st.session_state.keys():
        st.session_state.messages = [
            {"role": "assistant", "content": "upload Document and ask me a question"}]

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
