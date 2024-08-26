import streamlit as st
import torch
from transformers import AutoModelForTokenClassification, BertTokenizerFast, pipeline

# Load the fine-tuned model and tokenizer
model_name = "ner_model"
model = AutoModelForTokenClassification.from_pretrained(model_name)
tokenizer = BertTokenizerFast.from_pretrained("tokenizer")

# Load the NER pipeline
nlp = pipeline("ner", model=model, tokenizer=tokenizer)

# Streamlit app
st.title("Named Entity Recognition with a LEGAL-BERT model")

st.write(
    """
    This app uses a fine-tuned BERT model to perform Named Entity Recognition (NER) on your input text.
    """
)

# Input text
input_text = st.text_area("Enter text to analyze:", "Priscilla Licup is the President of Dunkin Donuts")

if st.button("Analyze"):
    # Perform NER
    ner_results = nlp(input_text)

    # Display results
    st.write("### NER Results")
    for result in ner_results:
        st.write(f"Entity: `{result['word']}`, Label: `{result['entity']}`, Score: `{result['score']:.4f}`")

# Optional: Display the tokenizer's tokens and model predictions
if st.checkbox("Show tokens and model predictions"):
    tokens = tokenizer.tokenize(input_text)
    st.write(f"Tokens: {tokens}")

    inputs = tokenizer(input_text, return_tensors="pt")
    outputs = model(**inputs).logits
    predictions = torch.argmax(outputs, dim=2)
    labels = [model.config.id2label[label_id] for label_id in predictions[0].tolist()]
    
    st.write(f"Predicted Labels: {labels}")
