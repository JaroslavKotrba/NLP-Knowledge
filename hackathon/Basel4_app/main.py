# TO RUN APP
# conda create -n basel4 python=3.11
# pip install fastapi uvicorn
# uvicorn main:app --reload

import os
from enum import Enum
from fastapi import Form
from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from openai import OpenAI
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.chains.llm import LLMChain
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# API
os.getcwd()
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# Serve static files (CSS, JavaScript)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize Jinja2 for templates
templates = Jinja2Templates(directory="templates")

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize embeddings
embeddings_model = OpenAIEmbeddings(
    model="text-embedding-ada-002", openai_api_key=os.getenv("OPENAI_API_KEY")
)


# Ticket categories
class TicketCategory(str, Enum):
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


# LOAD vector_store
vector_store = FAISS.load_local(
    "data/store_app",
    embeddings_model,
    allow_dangerous_deserialization=True,
)

# Retrieve top 5 relevant chunks
retriever = vector_store.as_retriever(search_kwargs={"k": 5})


def classify_ticket(reply: str) -> TicketCategory:
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
    query_embedding = embeddings_model.embed_query(query)
    reply_embedding = embeddings_model.embed_query(reply)
    similarity = cosine_similarity([query_embedding], [reply_embedding])[0][0]
    return max(0, min(1, similarity))


def process_basel4_query(query: str) -> Ticket:
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

    chat_model = ChatOpenAI(
        model_name="gpt-3.5-turbo",  # gpt-4 (double pricing)
        temperature=0.5,
        top_p=1,
        openai_api_key=os.getenv("OPENAI_API_KEY"),
    )
    llm_chain = LLMChain(llm=chat_model, prompt=prompt_template)
    stuff_chain = StuffDocumentsChain(
        llm_chain=llm_chain, document_variable_name="context"
    )
    qa_chain = RetrievalQA(combine_documents_chain=stuff_chain, retriever=retriever)

    llm_reply = qa_chain.invoke({"query": query})["result"]

    print("Reply created by AI!")
    return Ticket(
        reply=llm_reply,
        category=classify_ticket(llm_reply),
        confidence=calculate_confidence(query, llm_reply),
    )


@app.get("/")
def read_root(request: Request):
    """Serve the HTML interface."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/ask/")
async def ask_question(query: str = Form(...)):
    try:
        ticket = process_basel4_query(query)
        return {
            "reply": ticket.reply,
            "category": ticket.category.value,
            "confidence": ticket.confidence,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
