# FINAL PRODUCT

import instructor
from pydantic import BaseModel, Field
from openai import OpenAI
from enum import Enum
from pydantic import BaseModel, BeforeValidator
from typing_extensions import Annotated
from instructor import llm_validator
import os
from dotenv import load_dotenv

client = instructor.from_openai(OpenAI(api_key=os.getenv("OPENAI_API_KEY")))

# load_dotenv()
# client = instructor.from_openai(OpenAI())

# --------------------------------------------------------------
# Ticket System Example with Structured Output
# --------------------------------------------------------------


class TicketCategory(str, Enum):
    """Enumeration of categories for incoming tickets."""

    GENERAL = "general"
    ORDER = "order"
    BILLING = "billing"


class CustomerSentiment(str, Enum):
    """Enumeration of customer sentiment labels."""

    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"


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
    sentiment: CustomerSentiment  # setting categories sentiment


def process_ticket(query: str) -> Ticket:  # -> will return Ticket() structure
    reply = client.chat.completions.create(
        model="gpt-3.5-turbo",
        response_model=Ticket,
        max_retries=3,
        messages=[
            {
                "role": "system",
                "content": "Analyze the incoming customer message and predict the values for the ticket.",
            },
            {"role": "user", "content": query},
        ],
    )

    return reply


ticket = process_ticket("Hi there, I have a question about my bill. Can you help me?")
assert ticket.category == TicketCategory.BILLING

print("Category:", ticket.category.value)
print("Sentiment:", ticket.sentiment.value)
print("Confidence:", ticket.confidence)
print("Reply:", ticket.reply)
