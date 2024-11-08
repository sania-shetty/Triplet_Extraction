import streamlit as st
import spacy
import coreferee
import re
import nltk
from nltk.tokenize import sent_tokenize
import docx
import os
from tabulate import tabulate

nltk.download('punkt')

nlp = spacy.load('en_core_web_sm')
nlp.add_pipe('coreferee')

class CorefResolver:
    def __init__(self, file):
        self.file = file

    def coref(self):
        with open(self.file, "r") as read_file:
            dirty_text = read_file.read()
        self.dirty_text = dirty_text
        cor_text = nlp(dirty_text)
        self.cor_text = cor_text
        chains = cor_text._.coref_chains
        chains.print()
        self.chains = chains

    def replace_coref(self):
        dirty_text = self.dirty_text
        regular = self.dirty_text
        cor_text = self.cor_text
        chains = self.chains
        for i in range(len(chains)):
            resolve = chains.resolve(cor_text[chains[i][1][0]])
            for mention in range(1, len(chains[i])):
                dirty_text = dirty_text.replace(str(cor_text[chains[i][mention][0]]), str(resolve[0]))
                regular = re.sub(r'\b' + re.escape(str(cor_text[chains[i][mention][0]])) + r'\b', str(resolve[0]), regular, count=1)
        with open("coref.txt", "w") as cleaned:
            cleaned.write(dirty_text)
        with open("regular.txt", "w") as reg:
            reg.write(regular)

def read_docx_file(filepath):
    doc = docx.Document(filepath)
    text = []
    for paragraph in doc.paragraphs:
        text.append(paragraph.text)
    return '\n'.join(text)

def read_txt_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def tokenize_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

def tokenize_and_lemmatize(sentence):
    doc = nlp(sentence)
    lemmatized_sentence = ' '.join([token.lemma_ for token in doc])
    return lemmatized_sentence

def extract_subject_verb_object(sentence):
    doc = nlp(sentence)
    subject = ""
    verb = ""
    obj = ""

    for token in doc:
        if "subj" in token.dep_:
            subject = token.text
        if "VERB" in token.pos_:
            verb = token.lemma_
        if "obj" in token.dep_:
            obj = token.text

    return subject, verb, obj

def main(filepath):
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.docx':
        text = read_docx_file(filepath)
    elif ext == '.txt':
        text = read_txt_file(filepath)
    else:
        raise ValueError("Unsupported file format")
    
 
    temp_file_path = "temp_text.txt"
    with open(temp_file_path, "w") as temp_file:
        temp_file.write(text)
    
   
    coref_resolver = CorefResolver(temp_file_path)
    coref_resolver.coref()
    coref_resolver.replace_coref()
    
    with open("regular.txt", "r") as reg_file:
        resolved_text = reg_file.read()
    
    sentences = tokenize_sentences(resolved_text)
    
    results = []
    for i, sentence in enumerate(sentences):
        lemmatized_sentence = tokenize_and_lemmatize(sentence)
        
        subject, verb, obj = extract_subject_verb_object(lemmatized_sentence)
        
        if subject and verb and obj:
            results.append((f"Sentence {i+1}", subject, verb, obj))

    return results

# Streamlit App
st.title("Structurization of Unstructured data")
st.write("Upload a .txt or .docx file to perform coreference resolution and extract Subject-Verb-Object from each sentence.")

uploaded_file = st.file_uploader("Choose a file", type=["txt", "docx"])

if uploaded_file is not None:
    if uploaded_file.name.endswith('.txt'):
        text = uploaded_file.getvalue().decode("utf-8")
        with open("uploaded_text.txt", "w") as f:
            f.write(text)
    elif uploaded_file.name.endswith('.docx'):
        text = read_docx_file(uploaded_file)
        with open("uploaded_text.txt", "w") as f:
            f.write(text)
    
    coref_resolver = CorefResolver("uploaded_text.txt")
    coref_resolver.coref()
    coref_resolver.replace_coref()
    
    with open("regular.txt", "r") as reg_file:
        resolved_text = reg_file.read()
    
    st.subheader("Resolved Coreferences")
    st.write(resolved_text)
    
    sentences = tokenize_sentences(resolved_text)
    
    results = []
    for i, sentence in enumerate(sentences):
        lemmatized_sentence = tokenize_and_lemmatize(sentence)
        subject, verb, obj = extract_subject_verb_object(lemmatized_sentence)
        
        if subject and verb and obj:
            results.append((f"Sentence {i+1}", subject, verb, obj))

    if results:
        headers = ["Sentence", "Subject", "Verb", "Object"]
        table = tabulate(results, headers=headers, tablefmt='pipe')
        st.subheader("Extracted Subject-Verb-Object")
        st.text(table)
    else:
        st.write("No Subject-Verb-Object found in the text.")
