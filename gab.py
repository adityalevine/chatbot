import torch
import numpy as np
import xgboost as xgb
import json
import random
import streamlit as st
from transformers import BertTokenizer, BertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load intents dataset
def load_intents(file_path="KB.json"):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)["intents"]

data = load_intents()

# Extract patterns and tags
patterns, tags = [], []
tag_map = {}
for intent in data:
    for pattern in intent["patterns"]:
        patterns.append(pattern)
        tags.append(intent["tag"])
    tag_map[intent["tag"]] = len(tag_map)

# Convert tags to numeric labels
label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(tags)

# Load pre-trained BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
bert_model = BertModel.from_pretrained("bert-base-uncased")

def get_bert_embeddings(text_list):
    """Generate BERT embeddings for a list of texts."""
    embeddings = []
    for text in text_list:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = bert_model(**inputs)
            pooled_output = torch.mean(outputs.last_hidden_state, dim=1).squeeze().numpy()
            embeddings.append(pooled_output)
    return np.array(embeddings)

# Precompute BERT embeddings for all patterns (done only once)
X = get_bert_embeddings(patterns)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)

# Ensure all classes are present in y_train
missing_classes = set(range(len(tag_map))) - set(np.unique(y_train))
for missing_class in missing_classes:
    X_train = np.vstack([X_train, np.zeros((1, X_train.shape[1]))])
    y_train = np.append(y_train, missing_class)

# Train XGBoost model (done once during training)
xgb_model = xgb.XGBClassifier(
    objective='multi:softmax',
    num_class=len(tag_map),
    learning_rate=0.2,
    n_estimators=100,
    max_depth=10,
    eval_metric='mlogloss'
)
xgb_model.fit(X_train, y_train)

# Caching the precomputed BERT embeddings for faster inference
@st.cache_resource
def cached_embeddings():
    return X, label_encoder

# Chatbot response generation function
def chatbot_response(user_input):
    """Generate chatbot response based on user input."""
    # Get embeddings for user input
    user_embedding = get_bert_embeddings([user_input])
    # Predict the label
    predicted_label = xgb_model.predict(user_embedding)
    predicted_tag = label_encoder.inverse_transform(predicted_label)[0]
    
    for intent in data:
        if intent["tag"] == predicted_tag:
            return random.choice(intent["responses"])
    return "Maaf, saya tidak mengerti apa yang Anda maksud."

# Streamlit UI
st.title("Chatbot Berbasis AI")
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ketik pesan Anda di sini..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response after getting input
    response = chatbot_response(prompt)
    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
