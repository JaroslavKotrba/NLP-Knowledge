import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from enum import Enum
from pydantic import BaseModel, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator
import os

client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))


def load_confluence_text(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        return file.read()


class TicketCategory(str, Enum):
    """Enumeration of categories for incoming tickets."""

    GENERAL = "general"
    RISK = "risk"
    DEVELOPMENT = "development"


class Ticket(BaseModel):
    reply: Annotated[  # content filtering
        str,
        BeforeValidator(
            llm_validator(
                statement="Never say things that could hurt the reputation of the company.",
                client=client,
                allow_override=True,
            )
        ),
    ]
    category: TicketCategory  # setting categories ticket
    confidence: float = Field(  # setting confidence
        ge=0, le=1, description="Confidence in the category prediction."
    )  # ge=0 greateer than 0 le=1less than 1


def process_confluence_ticket(query: str, confluence_text: str) -> Ticket:
    prompt = f"Here is the relevant information from Confluence:\n{confluence_text}\n\nNow, based on this, answer the following query:\n{query}"

    reply = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Ticket,
        max_retries=3,
        messages=[
            {
                "role": "system",
                "content": "Analyze the following text and respond to the user query accordingly.",
            },
            {"role": "user", "content": prompt},
        ],
    )

    return reply


if __name__ == "__main__":
    # Load the Confluence text file
    confluence_text = load_confluence_text("../hackathon/confluence.txt")

    # Query to be answered
    query = "What is abreviation for CAR shortly explain?"

    # Process the ticket with the loaded Confluence data
    ticket = process_confluence_ticket(query, confluence_text)

    # Output the results
    print("Category:", ticket.category.value)
    print("Confidence:", ticket.confidence)
    print("Reply:", ticket.reply)
