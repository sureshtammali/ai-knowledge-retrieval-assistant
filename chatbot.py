import cohere
import uuid

class Chatbot:
    def __init__(self, vectorstore, cohere_api_key: str):
        self.vectorstore = vectorstore
        self.co = cohere.ClientV2(cohere_api_key)

    def respond(self, user_message: str):
        retrieved_docs = self.vectorstore.retrieve(user_message)
        messages = [{"role": "user", "content": user_message}]
        if retrieved_docs:
            messages.insert(0, {"role": "system", "content": "Use the following documents to answer:\n" + "\n".join(d['text'] for d in retrieved_docs)})
        response = self.co.chat_stream(
            model="command-r-plus-08-2024",
            messages=messages,
        )
        return response, retrieved_docs
