import os
from enum import Enum
from typing import List
from pydantic import BaseModel, Field
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
import pinecone
from pinecone import Pinecone, PodSpec

# Initialize OpenAI client
os.getcwd()
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load OpenAI embedding model to create vector store
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002",  # default model
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)


# Load pdf
def load_pdf_document(file_path: str) -> List[Document]:
    pdf_reader = PdfReader(file_path)
    text_content = ""
    for page in pdf_reader.pages:
        text_content += page.extract_text() + "\n"

    # Split the text into smaller chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    )
    texts = text_splitter.split_text(text_content)

    return [
        Document(page_content=t, metadata={"source": os.path.basename(file_path)})
        for t in texts
    ]


# Load pdfs to create embeddings
def load_multiple_pdfs(file_paths: List[str]) -> List[Document]:
    documents = []
    for file_path in file_paths:
        document = load_pdf_document(file_path)
        documents.extend(document)
    return documents


pdf_files = [  # add new documents here
    "Basel4_consul.pdf",
]

documents = load_multiple_pdfs(pdf_files)

# Initialize Pinecone
pinecone = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))


# SAVE
def create_or_update_pinecone_index(
    documents: List[Document], index_name: str, batch_size: int = 100
):
    # Check if the index exists, if not, create it
    if index_name not in pinecone.list_indexes():
        pinecone.create_index(
            name=index_name,
            dimension=1536,
            metric="cosine",
            spec=PodSpec(environment="gcp-starter"),
        )

    # Create the vector store
    vector_store = PineconeVectorStore.from_documents(
        documents,
        embeddings_model,
        index_name=index_name,
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
    )

    # Process documents in batches
    for i in range(0, len(documents), batch_size):
        batch = documents[i : i + batch_size]

        # Add documents to the vector store
        vector_store.add_documents(batch)

        print(f"Processed batch {i//batch_size + 1}/{len(documents)//batch_size + 1}")

    return vector_store


vector_store = create_or_update_pinecone_index(documents, "basel4-nlp")


# LOAD
def get_pinecone_vector_store(index_name: str) -> PineconeVectorStore:
    # Check if the index exists
    if index_name not in pinecone.list_indexes():
        raise ValueError(f"Index '{index_name}' does not exist in Pinecone.")

    # Get the Pinecone index
    index = pinecone.Index(index_name)

    # Create and return the vector store
    return PineconeVectorStore(index, embeddings_model.embed_query, "text")


vector_store = get_pinecone_vector_store("Basel4-index")


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
    )  # ge=0 greateer than 0 le=1less than 1


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
    query_embedding = embeddings_model.embed_query(query)  # from the input
    reply_embedding = embeddings_model.embed_query(reply)  # created by the model
    similarity = cosine_similarity([query_embedding], [reply_embedding])[0][0]
    return max(0, min(1, similarity))


def process_basel4_query(query: str) -> Ticket:

    # Template of the promt structure
    prompt_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an expert on Basel 4 regulations. Use the following pieces of context to answer the human's question about Basel 4.",
            ),
            ("human", "Context: {context}\n\nQuestion about Basel 4: {question}"),
            ("human", "Answer:"),
        ]
    )

    # Model definition
    chat_model = ChatOpenAI(
        model_name="gpt-4o-mini",  # gpt-3.5-turbo gpt-4o-mini
        temperature=0.5,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )

    # LLM chain
    llm_chain = LLMChain(llm=chat_model, prompt=prompt_template)

    # Docu chain
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="context"
    )

    # Model + 5 most relevant chunks
    qa_chain = RetrievalQA(combine_documents_chain=stuff_chain, retriever=retriever)

    # Result
    llm_reply = qa_chain.invoke({"query": query})["result"]

    # Debug
    print("Reply created by AI!")

    # Create a ticket with the response and dynamically determined metadata
    ticket = Ticket(
        reply=llm_reply,
        category=classify_ticket(llm_reply),
        confidence=calculate_confidence(query, llm_reply),
    )

    return ticket


if __name__ == "__main__":
    # Example query about Basel 4
    query = "What are Imparments?"
    # query = "What are Securitisation exposures?"

    # Process the query using the RAG approach
    ticket = process_basel4_query(query)

    # Output the results
    print("Category:", ticket.category.value)
    print("Confidence:", ticket.confidence)
    print("Reply:", ticket.reply)
