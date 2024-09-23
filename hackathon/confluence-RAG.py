import os
from enum import Enum
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from sklearn.metrics.pairwise import cosine_similarity

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load OpenAI embedding model to create vector store
embeddings = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))


# Function to load and retrieve documents
def load_confluence_text(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


# Load and embed the Confluence document
confluence_text = load_confluence_text("../hackathon/confluence.txt")
documents = [Document(page_content=confluence_text, metadata={"source": "Confluence"})]

# Create FAISS vector store and retriever
vector_store = FAISS.from_documents(documents, embeddings)
retriever = vector_store.as_retriever()


# Define ticket categories
class TicketCategory(str, Enum):
    """Enumeration of categories for incoming tickets."""

    GENERAL = "general"
    RISK = "risk"
    DEVELOPMENT = "development"


# Ticket model to hold the reply and associated metadata
class Ticket(BaseModel):
    reply: str
    category: TicketCategory
    confidence: float = Field(
        ge=0, le=1, description="Confidence in the category prediction."
    )


def classify_ticket(reply: str) -> TicketCategory:
    """Classify the ticket based on keywords present in the reply."""
    reply_lower = reply.lower()
    risk_keywords = [
        "risk",
        "credit",
        "capital",
        "exposure",
        "loss",
        "limit",
        "stress testing",
    ]
    development_keywords = [
        "develop",
        "development",
        "code",
        "implementation",
        "programming",
        "deployment",
    ]

    if any(keyword in reply_lower for keyword in risk_keywords):
        return TicketCategory.RISK
    elif any(keyword in reply_lower for keyword in development_keywords):
        return TicketCategory.DEVELOPMENT
    else:
        return TicketCategory.GENERAL


def calculate_confidence(query: str, reply: str) -> float:
    """Calculate the confidence score dynamically based on query and reply."""
    query_embedding = embeddings.embed_query(query)
    reply_embedding = embeddings.embed_query(reply)
    similarity = cosine_similarity([query_embedding], [reply_embedding])[0][0]
    return max(0, min(1, similarity))


def process_confluence_ticket(query: str) -> Ticket:
    # Create a chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Use the following pieces of context to answer the human's question. If you don't know the answer, just say that you don't know, don't try to make up an answer.",
            ),
            ("human", "Context: {context}\n\nQuestion: {question}"),
            ("human", "Answer:"),
        ]
    )

    # Initialize ChatOpenAI model
    chat_model = ChatOpenAI(
        model_name="gpt-3.5-turbo", openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # Create an LLMChain
    llm_chain = LLMChain(llm=chat_model, prompt=prompt_template)

    # Create a StuffDocumentsChain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="context"
    )

    # Create a RetrievalQA chain
    qa_chain = RetrievalQA(
        combine_documents_chain=stuff_chain,
        retriever=retriever,
    )

    # Run the chain with the query
    llm_reply = qa_chain.invoke({"query": query})["result"]

    # Create a ticket with the response and dynamically determined metadata
    ticket = Ticket(
        reply=llm_reply,
        category=classify_ticket(llm_reply),
        confidence=calculate_confidence(query, llm_reply),
    )

    return ticket


if __name__ == "__main__":
    # Example query to be answered
    query = "What is abbreviation for CAR? Explain briefly."

    # Process the query using the RAG approach
    ticket = process_confluence_ticket(query)

    # Output the results
    print("Category:", ticket.category.value)
    print("Confidence:", ticket.confidence)
    print("Reply:", ticket.reply)
