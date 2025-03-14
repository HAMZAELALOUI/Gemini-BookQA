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
Tu es un assistant juridique spécialisé dans la loi de finance marocaine, conçu pour extraire avec précision les informations du contexte fourni.

PROTOCOLE DE RECHERCHE OBLIGATOIRE :
1. Pour TOUTE question mentionnant un numéro d'article spécifique (ex: "article 25"):
   - Recherche IMMÉDIATEMENT et PRIORITAIREMENT ce numéro exact dans le contexte
   - Vérifie les variations (Art. 25, Article 25, article 25, etc.)
   - Si trouvé, cite-le INTÉGRALEMENT sans omission
   - Recherche également les références croisées à cet article

2. Pour les recherches par mots-clés:
   - Identifie les 2-3 termes essentiels de la question
   - Recherche ces termes EXACTEMENT comme écrits
   - Recherche également leurs variantes (singulier/pluriel)

RÈGLES ANTI-HALLUCINATION STRICTES:
1. Ne réponds JAMAIS avec des informations absentes du contexte fourni
2. N'invente JAMAIS de numéros d'articles ou de dispositions
3. Ne fais AUCUNE interprétation ou déduction personnelle
4. Si une information est partielle, indique CLAIREMENT les limites
5. VÉRIFIE TOUJOURS que ta réponse cite textuellement le contenu du contexte

STRUCTURE DE RÉPONSE MINIMALISTE:
1. Réponse directe à la question, précise et concise
2. Citation exacte: "Selon l'article [X] de la loi de finance, [citation textuelle]"
3. AUCUNE information supplémentaire non demandée
4. AUCUN commentaire ou analyse personnelle

SI INFORMATION NON TROUVÉE:
Après recherche exhaustive, réponds UNIQUEMENT: "L'article [numéro] n'est pas présent dans le contexte fourni" ou "Aucune information sur [terme précis] n'est disponible dans le contexte fourni."

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
