import os
import sqlite3
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, Trainer, TrainingArguments, pipeline
import faiss
import numpy as np
import fitz
import pymupdf
import time
from datasets import Dataset
from langchain.schema import Document
from docx import Document as DocxDocument


class RAGRetriever:
    def __init__(self, db_path="training_logs.db", model_name='./ftrag_model',
                 embedding_model_name="sentence-transformers/all-MiniLM-L6-v2",
                 summarization_model_name="facebook/bart-large-cnn"): # meta-Llama/Llama-3.2-1B-Instruct
        self.db_path = db_path
        self.texts = []
        self.index = None
        self.ft_data = []

        # Load tokenizer and models
        self.tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
        self.embedding_model = AutoModel.from_pretrained(embedding_model_name)
        self.llama_tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llama_model = AutoModelForCausalLM.from_pretrained(model_name)

        # Load summarization model
        self.summarizer = pipeline("summarization", model=summarization_model_name)

        # Set pad token if not already set
        if self.tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token or "[PAD]"
            self.llama_model.resize_token_embeddings(len(self.llama_tokenizer))

        # Initialize logging database
        self._initialize_log_db()

    def _initialize_log_db(self):
        """Initialize logging database for training metrics."""
        self.conn = sqlite3.connect(self.db_path)
        self.cursor = self.conn.cursor()
        self.cursor.execute('''
        CREATE TABLE IF NOT EXISTS training_metrics (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            start_time TEXT,
            end_time TEXT,
            epochs INTEGER,
            loss REAL,
            accuracy REAL
        )
        ''')
        self.conn.commit()

    def load_text_from_files(self, folder_path):
        """Helper method to load text from all PDFs and DOCX files in a folder and its subfolders."""
        texts = []
        for root, _, files in os.walk(folder_path):  # Walk through all subfolders
            for filename in files:
                file_path = os.path.join(root, filename)
                try:
                    if filename.endswith(".pdf"):
                        texts.append(self._extract_text_from_pdf(file_path))
                    elif filename.endswith(".docx"):
                        texts.append(self._extract_text_from_docx(file_path))
                except Exception as e:
                    print(f"Error reading {file_path}: {e}")
        return texts

    def _extract_text_from_pdf(self, pdf_path):
        """Extract text from a PDF file using PyMuPDF."""
        text = ""
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
        return text

    def _extract_text_from_docx(self, docx_path):
        """Extract text from a DOCX file."""
        text = ""
        doc = DocxDocument(docx_path)
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text

    def load_ft_data(self, folder_path="ft_file"):
        """Load fine-tuning data from PDFs and DOCX files in `ft_file` folder and its subfolders."""
        self.ft_data = self.load_text_from_files(folder_path)
        print(f"Loaded fine-tuning data from {len(self.ft_data)} files in {folder_path} and its subfolders.")

    def load_rag_data(self, folder_path="rag_file"):
        """Load RAG data from PDFs and DOCX files in the `rag_file` folder and its subfolders."""
        self.texts = self.load_text_from_files(folder_path)
        print(f"Loaded RAG data from {len(self.texts)} files in {folder_path}.")

    def create_embeddings(self, batch_size=8):
        """Create embeddings for all texts and store them in a faiss index."""
        if not self.texts: # Check if there any texts to process
            print("No texts available to create embeddings.")
            return

        embeddings = []
        for i in range(0, len(self.texts), batch_size):
            batch_texts = self.texts[i:i + batch_size]
            inputs = self.tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True)
            with torch.no_grad():
                batch_embeddings = self.embedding_model(**inputs).last_hidden_state.mean(dim=1).numpy()
            embeddings.append(batch_embeddings)

        if embeddings: # Only concatenate if there are embeddings created
            embeddings = np.concatenate(embeddings, axis=0)
            self.index = faiss.IndexFlatL2(embeddings.shape[1])
            self.index.add(embeddings)
            print("Created FAISS index with embeddings.")
        else:
            print("No texts available to create embeddings.")

    def retrieve_context(self, query, k=5, relevance_threshold=0.2):
        """
        Retrieve top-k relevant contexts for a given query and summarize them if applicable.
        :param query: The input query string.
        :param k: The number of documents to retrieve.
        :param relevance_threshold: The similarity threshold to filter out non-relevant results.
        :return: Combined summarized context or a fallback message.
        """
        # Compute query embedding
        query_embedding = self.embedding_model(
            **self.tokenizer(query, return_tensors="pt")
        ).pooler_output.detach().numpy()

        # Perform FAISS search
        distances, indices = self.index.search(query_embedding, k)

        # Retrieve and filter documents based on relevance threshold
        retrieved_docs = []
        for i, distance in enumerate(distances[0]):
            if distance > relevance_threshold:  # Check similarity threshold
                break
            retrieved_docs.append(self.texts[indices[0][i]])

        # Handle case with no relevant documents
        if not retrieved_docs:
            return "No relevant data was found for this query."

        # Summarize each retrieved document
        summarized_docs = []
        for doc in retrieved_docs:
            try:
                input_length = len(self.llama_tokenizer(doc)["input_ids"])
                max_length = max(10, min(100, int(input_length * 0.3)))
                min_length = max(5, int(max_length * 0.5))

                summary = self.summarizer(doc, max_length=max_length, min_length=min_length, do_sample=False)
                summarized_docs.append(summary[0]["summary_text"])
            except Exception as e:
                print(f"Error summarizing document: {e}")
                summarized_docs.append(doc)  # Use full document if summarization fails

        # Combine summarized documents
        combined_context = " ".join(summarized_docs)
        return combined_context

    def generate_response(self, query, max_new_tokens=100):
        """
        Generate a response based on retrieved and summarized context or fallback to query-only generation.
        """
        # Retrieve summarized context
        context = self.retrieve_context(query, k=5)  # k is handled here, not passed to generate_response

        # Use query alone if no relevant context is found
        if context == "No relevant data was found for this query.":
            input_text = query
        else:
            input_text = f"{context}\n\n{query}"

        # Tokenize input and generate response
        inputs = self.llama_tokenizer(input_text, return_tensors="pt", truncation=True)
        output = self.llama_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.llama_tokenizer.pad_token_id  # Ensure padding token is set
        )
        return self.llama_tokenizer.decode(output[0], skip_special_tokens=True)

    def fine_tune(self, epochs=3):
        # Check if there is any fine-tuning data available
        if not self.ft_data:
            print("No fine-tuning data available. Please load data before fine-tuning.")
            return

            # Ensure padding token is set
            # Ensure padding token is set
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.eos_token or "[PAD]"

        """Fine-tune Llama model on loaded fine-tuning data."""
        tokenized_data = self.llama_tokenizer(self.ft_data,
                                              return_tensors="pt",
                                              padding=True,
                                              truncation=True,
                                              max_length=512)

        # Use `input_ids` as `labels` for language modeling
        tokenized_data["labels"] = tokenized_data["input_ids"].clone()

        # Convert to dataset format
        ft_dataset = Dataset.from_dict(tokenized_data)

        training_args = TrainingArguments(
            output_dir="./rag_llama",
            evaluation_strategy="no",
            per_device_train_batch_size=1,
            #per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            num_train_epochs=epochs,
            save_steps=500,
            save_total_limit=2,
            logging_dir="./logs",
            #fp16=True,
        )

        trainer = Trainer(
            model=self.llama_model,
            args=training_args,
            train_dataset=ft_dataset,
        )

        start_time = time.time()
        trainer.train()
        end_time = time.time()
        self._log_metrics(start_time, end_time, epochs)
        print("Fine-tuning completed and metrics logged.")

    def _log_metrics(self, start_time, end_time, epochs, loss=None, accuracy=None):
        """Log training metrics to the database."""
        self.cursor.execute('''
            INSERT INTO training_metrics (start_time, end_time, epochs, loss, accuracy)
            VALUES (?, ?, ?, ?, ?)
        ''', (time.ctime(start_time), time.ctime(end_time), epochs, loss, accuracy))
        self.conn.commit()

    def close_log_db(self):
        """Close the logging database connection."""
        self.conn.close()
        print("Log database connection closed.")


#model_name = 'meta-Llama/Llama-3.2-1B-Instruct'

#access_token = os.getenv('llm_3_token')

# tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=access_token)
# model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=access_token)


# Initialize RAGRetriever instance
rag_retriever = RAGRetriever(model_name="./ftrag_model")

# # Load fine-tuning data from `ft_file` folder (supports both PDF and DOCX files)
# print("Loading fine-tuning data...")
# rag_retriever.load_ft_data("./ft_file")

# Load RAG data from `rag_file` folder for retrieval-augmented generation
print("Loading RAG data...")
rag_retriever.load_rag_data("./rag_file")

# Create embeddings for the loaded RAG data and build a FAISS index for retrieval
print("Creating embeddings and building FAISS index...")
rag_retriever.create_embeddings()

# # Fine-tune the model with the loaded fine-tuning data
# print("Starting fine-tuning process...")
# rag_retriever.fine_tune(epochs=3)
# print("Fine-tuning completed.")

# Perform evaluation using queries
def evaluate_rag_model(queries, top_k=5):
    """
    Evaluate the RAG model by generating responses to a list of queries.
    :param queries: List of queries for evaluation.
    :param top_k: Number of contexts to retrieve.
    """
    for query in queries:
        print(f"\nQuery: {query}")
        response = rag_retriever.generate_response(query)
        print(f"Response: {response}")

# Generate a response to a sample query using RAG
# query = "Describe the model training."
# print(f"\nGenerating a response to the query: '{query}'")
# response = rag_retriever.generate_response(query)
# print("Generated Response:", response)
#
# # Evaluate the model with sample queries
# evaluate_rag_model(query)

# # Close the logging database
# rag_retriever.close_log_db()
# print("Database connection closed.")
