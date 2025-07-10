import streamlit as st
import requests
import tempfile
import json

API_URL = "https://fictional-space-potato-pv756rxwg56frg4x-8000.app.github.dev"  # Use localhost for local dev

st.set_page_config(page_title="Code Embed App", layout="centered")
st.title("🧠 Codebase Semantic Search")

tab1, tab2 = st.tabs(["📤 Upload Codebase", "🔍 Search Code"])

# Upload ZIP and embed
with tab1:
    st.subheader("Upload .zip of Python files")
    uploaded_file = st.file_uploader("Upload your codebase", type=["zip"])

    if uploaded_file and st.button("🚀 Upload & Generate Embeddings"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        with open(tmp_path, "rb") as f:
            files = {"file": (uploaded_file.name, f, "application/zip")}
            try:
                with st.spinner("Embedding code..."):
                    res = requests.post(f"{API_URL}/upload", files=files)
                if res.status_code == 200:
                    st.success(res.json().get("message", "Upload succeeded"))
                else:
                    st.error(f"❌ Failed: {res.text}")
            except Exception as e:
                st.error(f"⚠️ Error: {e}")

# Search the embedded codebase
with tab2:
    st.subheader("Semantic Search")
    query = st.text_input("Type your query", placeholder="e.g. start web server")

    if query and st.button("🔍 Search"):
        try:
            with st.spinner("Searching..."):
                res = requests.post(
                    f"{API_URL}/search",
                    headers={"Content-Type": "application/json"},
                    data=json.dumps({"text": query})
                )
            if res.status_code == 200:
                results = res.json()
                if not results:
                    st.warning("No results found.")
                else:
                    for r in results:
                        st.markdown(f"**Score:** `{r['score']:.4f}`")
                        st.code(r['text'], language="python")
            else:
                st.error(f"Search failed: {res.text}")
        except Exception as e:
            st.error(f"⚠️ Error: {e}")
