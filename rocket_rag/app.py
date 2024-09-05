import os
import json
import streamlit as st

from io import StringIO
from openai import OpenAI, OpenAIError
from typing import List, Dict

CONFIG_DIR = '../config'
cfg = json.load(open(os.path.join(CONFIG_DIR, "configs.json")))
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

st.title("ROCKET-RAG for Fault Diagnosis")

# ======================
# Function to handle file uploads
def handle_uploaded_file(uploaded_file):
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        file_content = stringio.read()
        st.write("File Content:")
        st.write(file_content)

# File uploader for uploading text files
uploaded_file = st.file_uploader("Upload a file (txt)", type=["txt"])
if uploaded_file:
    handle_uploaded_file(uploaded_file)

# ======================
# Chat interface
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Enter your message:", key="user_input"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=cfg['GPT_MODEL'],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
            ],
            stream=True,
        )
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
