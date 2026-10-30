
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

st.set_page_config(
    page_title="Smart AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 Smart AI Chatbot Using LLM")
st.caption("Powered by a pretrained language model — no API key required.")

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"

@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto"
    )

    model.eval()
    return tokenizer, model

with st.spinner("Loading the AI model..."):
    tokenizer, model = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
user_input = st.chat_input("Ask me anything...")

if user_input:
    st.session_state.messages.append(
        {"role": "user", "content": user_input}
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    prompt_messages = [
        {
            "role": "system",
            "content": (
                "You are a friendly AI assistant. "
                "Answer in simple, clear language."
            )
        }
    ]

    # Keep recent conversation context
    prompt_messages.extend(st.session_state.messages[-8:])

    prompt = tokenizer.apply_chat_template(
        prompt_messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=2048
    )

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=200,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )

            new_tokens = output[0][inputs["input_ids"].shape[1]:]
            answer = tokenizer.decode(
                new_tokens,
                skip_special_tokens=True
            ).strip()

            st.markdown(answer)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

if st.sidebar.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun()

st.sidebar.header("About")
st.sidebar.write(
    "This chatbot uses a pretrained LLM to understand "
    "your prompt and generate text responses."
)
