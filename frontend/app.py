import requests
import streamlit as st

API_URL="http://127.0.0.1:8000"

st.set_page_config(
    page_title="RAGDocu",
    layout="wide"
)

st.title("RAGDocu")
st.caption("Ask questions about your documents to AI")

if "messages" not in st.session_state:
    st.session_state.messages = []
    
# Sidebar: Document upload
with st.sidebar:
    st.header("Document Management")

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"]
    )

    if st.button(
        "Upload Document",
        disabled=uploaded_file is None,
        use_container_width=True
    ):
        try:
            with st.spinner("Processing document..."):
                response = requests.post(
                    f"{API_URL}/documents/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf"
                        )
                    },
                    timeout=180
                )

                response.raise_for_status()
                result = response.json()

            st.success(result["message"])

            st.write(f"**Filename:** {result['filename']}")
            st.write(f"**Chunks created:** {result['chunks_created']}")

        except requests.RequestException as error:
            st.error(f"Upload failed: {error}")

    st.divider()

    top_k = st.slider(
        "Number of retrieved chunks (Top-K)",
        min_value=1,
        max_value=10,
        value=3
    )

    if st.button("Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message["role"] == "assistant":
            sources = message.get("sources", [])

            if sources:
                with st.expander("View Sources"):
                    for index, source in enumerate(sources, start=1):
                        metadata = source["metadata"]

                        st.markdown(f"**Source {index}**")
                        st.write(f"Page: {metadata.get('page', '-')}")
                        st.write(f"Content: {source['text']}")
                        st.divider()


# Chat input
question = st.chat_input("Ask something about your documents...")

if question:
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Thinking..."):
                response = requests.post(
                    f"{API_URL}/chat/",
                    json={
                        "question": question,
                        "top_k": top_k
                    },
                    timeout=180
                )

                response.raise_for_status()
                result = response.json()

            answer = result["answer"]
            sources = result.get("sources", [])

            st.markdown(answer)

            if sources:
                with st.expander("View Sources"):
                    for index, source in enumerate(sources, start=1):
                        metadata = source["metadata"]

                        st.markdown(f"**Source {index}**")
                        st.write(f"Page: {metadata.get('page', '-')}")
                        st.write(f"Content: {source['text']}")
                        st.divider()

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources
            })

        except requests.RequestException as error:
            error_message = f"Failed to communicate with API: {error}"
            st.error(error_message)

            st.session_state.messages.append({
                "role": "assistant",
                "content": error_message,
                "sources": []
            })