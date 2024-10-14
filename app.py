from langchain.llms import OpenAI
import streamlit as st
import os
import json

# Function to load OpenAI model and get response
@st.cache_data(show_spinner=True)
def get_openai_response(question, context, model_name="text-davinci-003", temperature=0.5):
    llm = OpenAI(model_name=model_name, temperature=temperature)
    complete_input = f"{context}\nUser: {question}\nAI:"
    response = llm(complete_input)
    return response.strip()

# Function to save conversation history
def save_conversation(history):
    with open("conversation_history.json", "w") as f:
        json.dump(history, f)

# Function to load conversation history
def load_conversation():
    try:
        with open("conversation_history.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return []

# Initialize Streamlit app
st.set_page_config(page_title="Advanced Q&A Chatbot", page_icon="🤖", layout="centered")

st.title("Advanced Q&A Chatbot Demo")
st.write("Engage in a conversation, and receive responses from OpenAI's language model!")

# Load conversation history
conversation_history = load_conversation()

# User input and settings
with st.form(key="input_form"):
    user_input = st.text_input("Enter your question:", key="input")
    
    model_name = st.selectbox(
        "Choose a model:",
        ["text-davinci-003", "gpt-3.5-turbo", "gpt-4"]
    )
    
    temperature = st.slider(
        "Response randomness (temperature):", min_value=0.0, max_value=1.0, value=0.5
    )
    
    submit_button = st.form_submit_button(label="Ask")
    
    if st.button("Save Conversation"):
        save_conversation(conversation_history)
        st.success("Conversation saved!")

# Handle the response
if submit_button:
    if user_input.strip() == "":
        st.error("Please enter a valid question.")
    else:
        with st.spinner("Fetching the response..."):
            try:
                # Build context from conversation history
                context = "\n".join(conversation_history)
                response = get_openai_response(user_input, context, model_name, temperature)
                
                # Update conversation history
                conversation_history.append(f"User: {user_input}")
                conversation_history.append(f"AI: {response}")

                st.subheader("Response:")
                st.write(response)
                
                # Display conversation history
                if conversation_history:
                    st.subheader("Conversation History:")
                    for message in conversation_history:
                        st.write(message)
            except Exception as e:
                st.error(f"An error occurred: {e}")

