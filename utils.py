import numpy as np


def retrieve_context(query, top_k=3):
    # Tokenize query and generate embeddings
    inputs = tokenizer(query, return_tensors="pt")
    query_embedding = model(**inputs).last_hidden_state.mean(1).detach().numpy()

    # Retrieve relevant documents from FAISS index
    _, indices = index.search(query_embedding, top_k)
    retrieved_docs = [documents[i] for i in indices[0]]

    # Combine retrieved documents into a single context
    context = "\n".join(retrieved_docs)
    return context


def generate_response(query, top_k=3, max_length=200):
    # Retrieve context using the RAG model
    context = retrieve_context(query, top_k)

    # Combine context with the user query
    input_text = f"Context: {context}\nQuery: {query}\nAnswer:"
    inputs = tokenizer(input_text, return_tensors="pt")

    # Generate response using the fine-tuned model
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        num_beams=5,
        early_stopping=True
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response
