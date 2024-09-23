# TRANSFORMERS

from transformers import pipeline


# --------------------------------------------------------------
# Sentiment analysis
# --------------------------------------------------------------

classifier = pipeline("sentiment-analysis")

# Input text
text = """
    Paris is the capital and most populous city of France. With an estimated population of 2,165,423 residents in 2019, within its administrative limits, Paris is also the most populous urban area in France and Europe. The metropolis of Paris is known as Paris Region (Île-de-France), which has an estimated population of 12,260,821 in 2019. 
    Situated on the Seine River, Paris is internationally famous for its museums, including the Louvre and the Musée d'Orsay; its landmarks, like the Eiffel Tower; and its historical neighborhoods, such as the Marais and Montmartre. Paris is one of the world's major centers for finance, commerce, fashion, science, and the arts. Paris is also known for its university and research institutions, such as the Sorbonne University and the École Polytechnique.
"""

res = classifier(text)

print(res)

# --------------------------------------------------------------
# Text generation
# --------------------------------------------------------------

generator = pipeline("text-generation", model="distilgpt2")

res = generator(
    "In this course we will teach you how to",
    max_length=30,
    truncation=True,
    num_return_sequences=2,
)

print(res)
