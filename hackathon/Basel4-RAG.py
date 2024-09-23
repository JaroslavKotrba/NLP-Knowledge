import os
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
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
os.getcwd()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load OpenAI embedding model to create vector store
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # default model
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


def load_pdf_document(file_path: str) -> List[Document]:
    pdf_reader = PdfReader(file_path)
    text_content = ""
    for page in pdf_reader.pages:
        text_content += page.extract_text() + "\n"

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len  # 1000, 200 characters
    )
    texts = text_splitter.split_text(text_content)

    return [Document(page_content=t, metadata={"source": "Basel4.pdf"}) for t in texts]


# Load and embed the Basel4 document
documents = load_pdf_document("Basel4.pdf")

# Create FAISS vector store and retriever
vector_store = FAISS.from_documents(documents, embeddings_model)
# vector_store.save_local("store")  # SAVE

# Retrieve top 5 relevant chunks
# vector_store = FAISS.load_local(
#     "store", embeddings_model, allow_dangerous_deserialization=True
# )  # LOAD
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


# Define ticket categories
class TicketCategory(str, Enum):
    """Enumeration of categories for incoming tickets."""

    GENERAL = "general"
    RISK = "risk"
    CAPITAL = "capital"
    COMPLIANCE = "compliance"


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
    risk_keywords = ["risk", "credit", "exposure", "loss", "limit", "stress testing"]
    capital_keywords = [
        "capital",
        "tier 1",
        "tier 2",
        "risk-weighted assets",
        "leverage ratio",
    ]
    compliance_keywords = [
        "compliance",
        "regulation",
        "supervisory",
        "reporting",
        "disclosure",
    ]

    if any(keyword in reply_lower for keyword in risk_keywords):
        return TicketCategory.RISK
    elif any(keyword in reply_lower for keyword in capital_keywords):
        return TicketCategory.CAPITAL
    elif any(keyword in reply_lower for keyword in compliance_keywords):
        return TicketCategory.COMPLIANCE
    else:
        return TicketCategory.GENERAL


def calculate_confidence(query: str, reply: str) -> float:
    """Calculate the confidence score dynamically based on query and reply."""
    query_embedding = embeddings_model.embed_query(query)
    reply_embedding = embeddings_model.embed_query(reply)
    similarity = cosine_similarity([query_embedding], [reply_embedding])[0][0]
    return max(0, min(1, similarity))


def process_basel4_query(query: str) -> Ticket:
    # Create a chat prompt template
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert on Basel 4 regulations. Use the following pieces of context to answer the human's question about Basel4.",
            ),
            ("human", "Context: {context}\n\nQuestion about Basel 4: {question}"),
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
    # Example query about Basel 4
    # query = "What are Credit Quality Steps?"
    query = "What are Imparments?"
    # query = "What are Securitisation exposures?"

    # Process the query using the RAG approach
    ticket = process_basel4_query(query)

    # Output the results
    print("Category:", ticket.category.value)
    print("Confidence:", ticket.confidence)
    print("Reply:", ticket.reply)
