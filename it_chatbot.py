# it_chatbot.py
# Azure OpenAI GPT-5 Customer Support Chatbot with Tools API
# Clean, validated for 2024-07-01-preview

import os, json, logging, streamlit as st
from openai import AzureOpenAI
from dotenv import load_dotenv
from datetime import datetime

# ----------------------------------------------------------------------
# 1. Setup logging
# ----------------------------------------------------------------------
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler("logs/chatbot_debug.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------
# 2. Environment variables
# ----------------------------------------------------------------------
load_dotenv()
AZURE_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
if not AZURE_API_KEY or not AZURE_ENDPOINT:
    st.error("❌ Missing Azure OpenAI credentials. Please set them in .env")
    st.stop()

# ----------------------------------------------------------------------
# 3. Azure client
# ----------------------------------------------------------------------
client = AzureOpenAI(
    api_key=AZURE_API_KEY,
    azure_endpoint=AZURE_ENDPOINT,
    api_version="2024-07-01-preview",
)

# ----------------------------------------------------------------------
# 4. Mock data
# ----------------------------------------------------------------------
faq_data = [
    {"question": "How can I reset my password?", "answer": "Go to Account Settings → Security → Reset Password."},
    {"question": "What is the refund policy?", "answer": "Refunds are available within 30 days of purchase."},
    {"question": "Do you offer international shipping?", "answer": "Yes, we ship worldwide."},
]
order_data = [
    {"order_id": "ORD123", "status": "Shipped", "eta": "2025-10-28"},
    {"order_id": "ORD124", "status": "Delivered", "eta": "2025-10-22"},
    {"order_id": "ORD125", "status": "Processing", "eta": "2025-10-27"},
]

# ----------------------------------------------------------------------
# 5. Tool functions
# ----------------------------------------------------------------------
def lookup_faq(query: str):
    for faq in faq_data:
        if query.lower() in faq["question"].lower() or query.lower() in faq["answer"].lower():
            return faq["answer"]
    return "Sorry, I couldn't find that information."

def get_order_status(order_id: str):
    for order in order_data:
        if order["order_id"].lower() == order_id.lower():
            return f"Order {order_id} is {order['status']} and will arrive by {order['eta']}."
    return f"Order ID {order_id} not found."

# ----------------------------------------------------------------------
# 6. Tool schema
# ----------------------------------------------------------------------
tools = [
    {
        "type": "function",
        "function": {
            "name": "lookup_faq",
            "description": "Retrieve an FAQ answer.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order_status",
            "description": "Retrieve order status by ID.",
            "parameters": {
                "type": "object",
                "properties": {"order_id": {"type": "string"}},
                "required": ["order_id"],
            },
        },
    },
]

# ----------------------------------------------------------------------
# 7. Conversation logging
# ----------------------------------------------------------------------
log_filename = f"logs/conversation_{datetime.now():%Y%m%d_%H%M%S}.txt"
def log_conversation(role, content):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_filename, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {role.capitalize()}: {content}\n")

# ----------------------------------------------------------------------
# 8. Streamlit UI
# ----------------------------------------------------------------------
st.set_page_config(page_title="Customer Support Chatbot", page_icon="💬")
st.title("💬 Customer Support Chatbot")
st.caption("Azure OpenAI Tools API • Multi-turn Context • Logging")

# ----------------------------------------------------------------------
# 9. Initialize session state
# ----------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are a polite, accurate customer support assistant."},
        {"role": "user", "content": "How can I reset my password?"},
        {"role": "assistant", "content": "You can reset it under Account Settings → Security."},
    ]
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []

# ----------------------------------------------------------------------
# 10. Chat interaction
# ----------------------------------------------------------------------
user_input = st.chat_input("Type your message...")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.display_messages.append({"role": "user", "content": user_input})
    log_conversation("user", user_input)

    try:
        # Step 1: Model initial reply
        completion = client.chat.completions.create(
            model="gpt-5",  # replace with your deployment name if needed
            messages=st.session_state.messages,
            tools=tools
        )

        message = completion.choices[0].message
        tool_calls = getattr(message, "tool_calls", None)

        if tool_calls:
            # Step 2: Execute tool(s)
            for tc in tool_calls:
                func_name = tc.function.name
                func_args = json.loads(tc.function.arguments)
                if func_name == "lookup_faq":
                    result = lookup_faq(func_args.get("query", ""))
                elif func_name == "get_order_status":
                    result = get_order_status(func_args.get("order_id", ""))
                else:
                    result = "Tool not implemented."

                # Step 3: Follow-up model call with tool result
                follow_up_messages = st.session_state.messages + [
                    message,  # include model's tool call
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result
                    }
                ]
                second_completion = client.chat.completions.create(
                    model="gpt-5",
                    messages=follow_up_messages,
                )
                final_msg = second_completion.choices[0].message
                reply = final_msg.content or result

                st.session_state.messages.append({"role": "assistant", "content": reply})
                st.session_state.display_messages.append({"role": "assistant", "content": reply})
                log_conversation("assistant", reply)

        else:
            # No tool calls, normal text reply
            reply = message.content or "No response."
            st.session_state.messages.append({"role": "assistant", "content": reply})
            st.session_state.display_messages.append({"role": "assistant", "content": reply})
            log_conversation("assistant", reply)

    except Exception as e:
        st.error(f"❌ API Error: {e}")
        logger.exception("API Error")
        log_conversation("error", str(e))

# ----------------------------------------------------------------------
# 11. Render conversation
# ----------------------------------------------------------------------
for msg in st.session_state.display_messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# ----------------------------------------------------------------------
# 12. Sidebar
# ----------------------------------------------------------------------
st.sidebar.markdown("### ℹ️ Instructions")
st.sidebar.info(
    "This chatbot demonstrates few-shot prompting, chain-of-thought reasoning, function calling (tool syntax), "
    "and timestamped conversation logs.\n\n"
)
st.sidebar.markdown("---")
st.sidebar.caption(f"🕓 Logs saved to `{log_filename}`")
