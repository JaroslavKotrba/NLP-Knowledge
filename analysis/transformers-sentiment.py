# TRANSFORMERS

from transformers import pipeline

classifier = pipeline("sentiment-analysis")

# Input text
text = """
    Paris is the capital and most populous city of France. With an estimated population of 2,165,423 residents in 2019, within its administrative limits, Paris is also the most populous urban area in France and Europe. The metropolis of Paris is known as Paris Region (Île-de-France), which has an estimated population of 12,260,821 in 2019. 
    Situated on the Seine River, Paris is internationally famous for its museums, including the Louvre and the Musée d'Orsay; its landmarks, like the Eiffel Tower; and its historical neighborhoods, such as the Marais and Montmartre. Paris is one of the world's major centers for finance, commerce, fashion, science, and the arts. Paris is also known for its university and research institutions, such as the Sorbonne University and the École Polytechnique.
"""

res = classifier(text)

print(res)

# GLOSSARY

# TOKENIZING
# Tokenizing the text (breaking it into smaller units like words or subwords), and converting it into a format suitable for training.
# Introduced by Vaswani et al. in 2017, the Transformer architecture is the backbone of most modern LLMs. It consists of layers of attention mechanisms and feed-forward neural networks.

# SELF-SUPERVISED LEARNING
# Self-supervised learning is a form of unsupervised learning where the model learns to predict part of the input from other parts of the input. This method allows the model to learn useful representations and patterns from the data without requiring manually labeled datasets.

# RETRIEVAL-AUGMENTED GENERATON (RAG)
# Retrieval-Augmented Generation (RAG) is a hybrid approach that enhances the capabilities of large language models (LLMs) by combining information retrieval with text generation.
# This method leverages the strengths of both retrieval systems and generative models to produce more accurate and contextually relevant responses.

# EMBEDDINGS
# Embeddings transform the query text into a numerical vector that captures the semantic meaning of the text.

# FINE-TUNING
# Fine-Tuning: Modifies the internal weights of the model to adapt it to a specific task. It requires labeled task-specific data for training.
# RAG: Enhances generation by retrieving relevant information from external sources at inference time. It uses retrieval to augment the generation process without changing the model’s internal weights.
