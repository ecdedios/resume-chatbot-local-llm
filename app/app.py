import os
import tempfile
import streamlit as st
from streamlit_chat import message
from rag import ChatCSV

# adds a title for the web page
st.set_page_config(page_title="Resume Chatbot")

def display_messages():
    """
    Displays chat messages in the Streamlit app.

    This function assumes that chat messages are stored in the Streamlit session state
    under the key "messages" as a list of tuples, where each tuple contains the message
    content and a boolean indicating whether it's a user message or not.

    Additionally, it creates an empty container for a thinking spinner in the Streamlit
    session state under the key "thinking_spinner".

    Note: Streamlit (st) functions are used for displaying content in a Streamlit app.
    """
    # Display a subheader for the chat.
    st.subheader("Chat")

    # Iterate through messages stored in the session state.
    for i, (msg, is_user) in enumerate(st.session_state["messages"]):
        # Display each message using the message function with appropriate styling.
        message(msg, is_user=is_user, key=str(i))

    # Create an empty container for a thinking spinner and store it in the session state.
    st.session_state["thinking_spinner"] = st.empty()

def process_input():
    """
    Processes user input and updates the chat messages in the Streamlit app.

    This function assumes that user input is stored in the Streamlit session state
    under the key "user_input," and the question-answering assistant is stored
    under the key "assistant."

    Additionally, it utilizes Streamlit functions for displaying a thinking spinner
    and updating the chat messages.

    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """
    # Check if there is user input and it is not empty.
    if st.session_state["user_input"] and len(st.session_state["user_input"].strip()) > 0:
        # Extract and clean the user input.
        user_text = st.session_state["user_input"].strip()

        # Display a thinking spinner while the assistant processes the input.
        with st.session_state["thinking_spinner"], st.spinner(f"Thinking"):
            # Ask the assistant for a response based on the user input.
            agent_text = st.session_state["assistant"].ask(user_text)

        # Append user and assistant messages to the chat messages in the session state.
        st.session_state["messages"].append((user_text, True))
        st.session_state["messages"].append((agent_text, False))

def read_and_save_file():
    """
    Reads and saves the uploaded file, performs ingestion, and clears the assistant state.

    This function assumes that the question-answering assistant is stored in the Streamlit
    session state under the key "assistant," and file-related information is stored under
    the key "file_uploader."

    Additionally, it utilizes Streamlit functions for displaying spinners and updating the
    assistant's state.

    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """
    # Clear the state of the question-answering assistant.
    st.session_state["assistant"].clear()

    # Clear the chat messages and user input in the session state.
    st.session_state["messages"] = []
    st.session_state["user_input"] = ""

    # Iterate through the uploaded files in the session state.
    for file in st.session_state["file_uploader"]:
        # Save the file to a temporary location and get the file path.
        with tempfile.NamedTemporaryFile(delete=False) as tf:
            tf.write(file.getbuffer())
            file_path = tf.name

        # Display a spinner while ingesting the file.
        with st.session_state["ingestion_spinner"], st.spinner(f"Ingesting {file.name}"):
            st.session_state["assistant"].ingest(file_path)
        os.remove(file_path)


def page():
    """
    Defines the content of the Streamlit app page for ChatCSV.

    This function sets up the initial session state if it doesn't exist and displays
    the main components of the Streamlit app, including the header, file uploader,
    and associated functionalities.

    Note: Streamlit (st) functions are used for interacting with the Streamlit app.
    """
    # Check if the session state is empty (first time loading the app).
    if len(st.session_state) == 0:
        # Initialize the session state with empty chat messages and a ChatCSV assistant.
        st.session_state["messages"] = []
        st.session_state["assistant"] = ChatCSV()

    # Display the main header of the Streamlit app.
    st.header("ChatCSV")

    # Display a subheader and a file uploader for uploading CSV files.
    st.subheader("Upload a csv file")
    st.file_uploader(
        "Upload csv",
        type=["csv"],
        key="file_uploader",
        on_change=read_and_save_file,
        label_visibility="collapsed",
        accept_multiple_files=True,
    )

    # Create an empty container for a spinner related to file ingestion
    # and store it in the Streamlit session state under the key "ingestion_spinner".
    st.session_state["ingestion_spinner"] = st.empty()

    # Display chat messages in the Streamlit app using the defined function.
    display_messages()

    # Display a text input field for user messages in the Streamlit app.
    # The input field has a key "user_input," and the on_change event triggers the
    # "process_input" function when the input changes.
    st.text_input("Message", key="user_input", on_change=process_input)

# Check if the script is being run as the main module.
if __name__ == "__main__":
    # Call the "page" function to set up and run the Streamlit app.
    page()