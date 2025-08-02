from transformers import MarianMTModel, MarianTokenizer
import torch
import streamlit as st

# Load French model and tokenizer
fr_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
fr_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr").to("cpu")

# Load Spanish model and tokenizer
es_tokenizer = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
es_model = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-es").to("cpu")

@st.cache_resource
def translate_fr(text):
    inputs = fr_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = fr_model.generate(**inputs, max_length=128)
    return fr_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)

@st.cache_resource
def translate_es(text):
    inputs = es_tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
    translated_tokens = es_model.generate(**inputs, max_length=128)
    return es_tokenizer.decode(translated_tokens[0], skip_special_tokens=True)


