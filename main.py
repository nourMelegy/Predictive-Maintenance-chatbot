from rag import retrieve_information
import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import string
from langchain_core.runnables import RunnableBranch, RunnableLambda
from langchain_core.tools import tool
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from langchain_groq import ChatGroq
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.2
)
class IntentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
    def forward(self, x):
        embedded = self.embedding(x)
        output, (hidden, cell) = self.lstm(embedded)
        logits = self.fc(hidden[-1])
        return logits
data = pd.read_csv("intents.csv")
X = data["text"]
y = data["intent"]
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp
)
def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.split()
vocab = {
    "<pad>": 0,
    "<unk>": 1
}

for sentence in X_train:
    for word in tokenize(sentence):
        if word not in vocab:
            vocab[word] = len(vocab)
def encode_sentence(sentence):
    return [vocab.get(word, vocab["<unk>"]) for word in tokenize(sentence)]
def pad_sequence(sequence, max_length):
    sequence = sequence[:max_length]
    return sequence + [vocab["<pad>"]] * (max_length - len(sequence))
max_length = max(len(encode_sentence(sentence)) for sentence in X_train)
model = IntentClassifier(
    vocab_size=len(vocab),
    embedding_dim=64,
    hidden_dim=64,
    num_classes=4
)
checkpoint = torch.load(
    "classifier.pt",
    weights_only=False
)
vocab = checkpoint["vocab"]
label_encoder = checkpoint["label_encoder"]
max_length = checkpoint["max_length"]
model = IntentClassifier(
    vocab_size=len(vocab),
    embedding_dim=64,
    hidden_dim=64,
    num_classes=len(label_encoder.classes_)
)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
print("Trained classifier loaded successfully!")
@tool
def maintenance_schedule():
    """Provides general vehicle maintenance schedule information."""
    return "General maintenance includes regular oil changes, tire checks, brake checks, fluid checks, and inspections."
@tool
def troubleshooting_guide():
    """Provides general troubleshooting information for common vehicle problems."""
    return "Check warning lights, unusual sounds, fluid levels, tire condition, and other visible symptoms."
@tool
def maintenance_tips():
    """Provides general preventive maintenance advice."""
    return "Regular inspections, correct fluid levels, tire maintenance, and following the vehicle maintenance schedule can help prevent problems."
@tool
def fault_check():
    """Provides general guidance for identifying possible vehicle faults."""
    return "Use the reported symptoms, warning lights, unusual sounds, smells, and vehicle behavior to identify possible faults."
def predict_intent(message):
    tokens = tokenize(message)
    encoded = [
        vocab.get(word, vocab["<unk>"])
        for word in tokens
    ]
    encoded = pad_sequence(encoded, max_length)
    input_tensor = torch.tensor(
        [encoded],
        dtype=torch.long
    )
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(
            probabilities,
            dim=1
        ).item()
        confidence = probabilities[0][predicted_class].item()
    intent = label_encoder.inverse_transform(
        [predicted_class]
    )[0]
    return intent, confidence
def build_agent(system_prompt, tools):
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}")
    ])
    agent = create_tool_calling_agent(
        llm,
        tools,
        prompt
    )
    return AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    return_intermediate_steps=True
    )
maintenance_advice = build_agent(
    """
    You are the Maintenance Advice Agent for a vehicle maintenance chatbot.

    Give practical preventive maintenance advice.

    Use the retrieved knowledge provided with the user's question.
    Base your answer on that retrieved knowledge when it is relevant.

    Do not invent specific maintenance intervals or vehicle specifications
    that are not supported by the retrieved knowledge.
    """,
    [maintenance_tips]
)

maintenance_prediction = build_agent(
    """
    You are the Maintenance Prediction Agent for a vehicle maintenance chatbot.

    Help users understand when maintenance may be needed.

    Use the retrieved knowledge provided with the user's question.
    Base your answer on that retrieved knowledge.

    If the retrieved knowledge does not contain a specific maintenance
    interval, tell the user to check the vehicle manufacturer's schedule
    instead of inventing an interval.

    Do not add extra maintenance advice, measurements, mileage,
    time intervals, symptoms, or technical details that are not present
    in the retrieved knowledge.
    """,
    [maintenance_schedule]
)

troubleshooting = build_agent(
    """
    You are the Troubleshooting Agent for a vehicle maintenance chatbot.

    Help users troubleshoot vehicle problems based on the symptoms they describe.

    Use the retrieved knowledge provided with the user's question.
    Base your answer on that retrieved knowledge.

    Do not invent technical details that are not supported by the retrieved knowledge.
    """,
    [troubleshooting_guide]
)

fault_diagnosis = build_agent(
    """
    You are the Fault Diagnosis Agent for a vehicle maintenance chatbot.

    Help users identify possible vehicle faults based on symptoms, warning lights,
    sounds, smells, and vehicle behavior.

    Use the retrieved knowledge provided with the user's question.
    Base your answer on that retrieved knowledge.

    Do not claim a definite diagnosis when the available information is not enough.
    Do not invent technical details that are not supported by the retrieved knowledge.
    """,
    [fault_check]
)
router = RunnableBranch(
    (
        lambda x: x["intent"] == "Maintenance Prediction",
        RunnableLambda(
            lambda x: maintenance_prediction.invoke({
                "input": (
                    x["message"]
                    + "\n\nRetrieved knowledge:\n"
                    + retrieve_information(x["message"])
                )
            })
        )
    ),
    (
        lambda x: x["intent"] == "Troubleshooting",
        RunnableLambda(
            lambda x: troubleshooting.invoke({
                "input": (
                    x["message"]
                    + "\n\nRetrieved knowledge:\n"
                    + retrieve_information(x["message"])
                )
            })
        )
    ),
    (
        lambda x: x["intent"] == "Maintenance Advice",
        RunnableLambda(
            lambda x: maintenance_advice.invoke({
                "input": (
                    x["message"]
                    + "\n\nRetrieved knowledge:\n"
                    + retrieve_information(x["message"])
                )
            })
        )
    ),
    (
        lambda x: x["intent"] == "Fault Diagnosis",
        RunnableLambda(
            lambda x: fault_diagnosis.invoke({
                "input": (
                    x["message"]
                    + "\n\nRetrieved knowledge:\n"
                    + retrieve_information(x["message"])
                )
            })
        )
    ),
    RunnableLambda(lambda x: "I could not determine the correct intent.")
)
st.title("Predictive Maintenance Chatbot")
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
if "last_intent" not in st.session_state:
    st.session_state.last_intent = "None"
if "last_confidence" not in st.session_state:
    st.session_state.last_confidence = 0.0
if "last_agent" not in st.session_state:
    st.session_state.last_agent = "None"
if "last_tool_calls" not in st.session_state:
    st.session_state.last_tool_calls = []
user_message = st.chat_input("Ask me about vehicle maintenance!")
if user_message:
    st.chat_message("user").write(user_message)

    st.session_state.messages.append({
        "role": "user",
        "content": user_message
    })

    intent, confidence = predict_intent(user_message)

    st.session_state.last_intent = intent
    st.session_state.last_confidence = confidence

    if intent == "Maintenance Prediction":
        st.session_state.last_agent = "Maintenance Prediction Agent"
    elif intent == "Troubleshooting":
        st.session_state.last_agent = "Troubleshooting Agent"
    elif intent == "Maintenance Advice":
        st.session_state.last_agent = "Maintenance Advice Agent"
    elif intent == "Fault Diagnosis":
        st.session_state.last_agent = "Fault Diagnosis Agent"

    conversation_history = "\n".join(
        [
            f"{message['role']}: {message['content']}"
            for message in st.session_state.messages
        ]
    )
    message_with_history = (
        "Conversation history:\n"
        + conversation_history
        + "\n\nCurrent user question:\n"
        + user_message
    )
    try:
        agent_result = router.invoke({
            "intent": intent,
            "message": message_with_history
        })

        result = agent_result["output"]

    except Exception as e:
        result = "Sorry, I could not process your request right now."
        st.error(f"Error: {e}")
    result = agent_result["output"]
    tool_calls = []

    for step in agent_result.get("intermediate_steps", []):
        action = step[0]
        tool_calls.append(action.tool)
    tool_calls.append("RAG Retrieval")
    st.chat_message("assistant").write(result)

    st.session_state.messages.append({
        "role": "assistant",
        "content": result
    })

    st.session_state.last_tool_calls = tool_calls
st.sidebar.write("Tool Calls:")

if st.session_state.last_tool_calls:
    for tool_name in st.session_state.last_tool_calls:
        st.sidebar.write(tool_name)
else:
    st.sidebar.write("None")
st.sidebar.title("Chatbot Information")
st.sidebar.write("Predicted Intent:")
st.sidebar.write(st.session_state.last_intent)
st.sidebar.write("Confidence:")
st.sidebar.write(f"{st.session_state.last_confidence:.2%}")
st.sidebar.write("Agent:")
st.sidebar.write(st.session_state.last_agent)
st.sidebar.write("Usage Counter:")
st.sidebar.write(len(st.session_state.messages) // 2)
