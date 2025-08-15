# ===== Imports =====
import os

import streamlit as st
from dotenv import load_dotenv, find_dotenv

# Embeddings / Vector store / Loader / Splitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# LLM + Chain
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

# PDF utils
from pypdf import PdfReader, PdfWriter  # Remplace PyPDF2
import base64
from tempfile import NamedTemporaryFile



# ====== Task 4: Process the Input PDF ======
def process_file(pdf_path: str):
    """Construit et retourne une ConversationalRetrievalChain prête à l’emploi sur le PDF."""
    # 1) Charger PAR PAGES
    docs = PyPDFLoader(pdf_path).load()

    # 2) Split en chunks (contexte robuste)
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    # 3) Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # 4) Base vectorielle
    # création manuelle du store
    vector_store = Chroma(
        collection_name="my_collection",
        embedding_function=embeddings,
        persist_directory="db"  # optionnel, pour stockage persistant
    )

    # ajout des documents chunkés
    vector_store.add_documents(chunks)


    # 5) LLM
    llm = ChatOpenAI(temperature=0.5)

    # 6) Chaîne conversationnelle (k élévé pour + d'anthropie)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(search_kwargs={"k": 6}),
        return_source_documents=True,
        verbose=False,
    )

    return chain


# ====== Utilitaires chat ======
def build_chat_history(messages):
    """Transforme st.session_state.messages -> [(user, assistant), ...]"""
    history, pending = [], None
    for m in messages:
        if m["role"] == "user":
            pending = m["content"]
        elif m["role"] == "assistant" and pending is not None:
            history.append((pending, m["content"]))
            pending = None
    return history


def ask_chain(query: str):
    """Appel la chain avec historique, renvoie (answer, pageNumber)."""
    history = build_chat_history(st.session_state.messages)
    resp = st.session_state.conversation({
        "question": query,
        "chat_history": history
    })
    answer = resp.get("answer", "")
    page_num = None
    src_docs = resp.get("source_documents", [])
    if src_docs:
        p = src_docs[0].metadata.get("page", None)
        if p is not None:
            page_num = int(p)
    return answer, page_num


# ====== App principale ======
def main():
    st.set_page_config(page_title="PDF RAG", page_icon="📄", layout="wide")
    load_dotenv()

    # State
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("conversation", None)
    st.session_state.setdefault("uploadedPDF", None)
    st.session_state.setdefault("pdf_path", None)
    st.session_state.setdefault("pageNumber", None)

    col1, col2 = st.columns(2)

    # ===== Colonne 1 : historique + upload/process =====
    with col1:
        st.header("PDF RAG")

        # Upload / Process
        with st.expander("Votre document", expanded=True):
            st.session_state.uploadedPDF = st.file_uploader("Choisir un PDF", type=["pdf"])
            if st.button("Process"):
                if st.session_state.uploadedPDF is None:
                    st.error("Charge un PDF avant stp.")
                else:
                    with NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                        st.session_state.uploadedPDF.seek(0)
                        tmp.write(st.session_state.uploadedPDF.getvalue())
                        tmp.flush()
                        st.session_state.pdf_path = tmp.name

                    with st.spinner("Indexation du PDF..."):
                        st.session_state.conversation = process_file(st.session_state.pdf_path)

                    # Message assistant initial
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "PDF indexé. Pose ta question en bas."
                    })
                    st.success("Prêt.")

    # ===== Colonne 2 : extrait PDF =====
    with col2:
        st.subheader("Extrait du PDF (page de référence ± 2)")
        if st.session_state.pdf_path and (st.session_state.pageNumber is not None):
            try:
                reader = PdfReader(st.session_state.pdf_path)
                N = int(st.session_state.pageNumber)  # 0-based
                start = max(N - 2, 0)
                end = min(N + 2, len(reader.pages) - 1)

                pdf_writer = PdfWriter()
                for i in range(start, end + 1):
                    pdf_writer.add_page(reader.pages[i])

                with NamedTemporaryFile(suffix=".pdf") as out_tmp:
                    with open(out_tmp.name, "wb") as out_f:
                        pdf_writer.write(out_f)
                    with open(out_tmp.name, "rb") as f:
                        base64_pdf = base64.b64encode(f.read()).decode("utf-8")

                page_to_focus = (N - start) + 1  # 1-based dans l'iframe
                iframe = f'''
                    <iframe
                        src="data:application/pdf;base64,{base64_pdf}#page={page_to_focus}"
                        width="100%" height="900" frameborder="0">
                    </iframe>
                '''
                st.markdown(iframe, unsafe_allow_html=True)
            except Exception as e:
                st.warning(f"Impossible d’afficher l’extrait PDF : {e}")
        else:
            st.info("Après une réponse référencée, l’extrait associé s’affichera ici.")

    # ===== Chat input  =====
    st.markdown("---")
    st.header("Votre discussion")
    user_query = st.chat_input("Pose ta question…")
    if user_query:
        if st.session_state.conversation is None:
            st.warning("Traite d’abord un PDF (bouton Process).")
        else:
            # Afficher tout l'historique
            for m in st.session_state.messages:
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
            # Afficher la question immédiatement
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.markdown(user_query)

            # Appeler la chain
            answer, page_num = ask_chain(user_query)

            # Afficher la réponse
            st.session_state.messages.append({"role": "assistant", "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)

            # Mémoriser la page si dispo
            if page_num is not None:
                st.session_state.pageNumber = page_num


if __name__ == "__main__":
    main()




