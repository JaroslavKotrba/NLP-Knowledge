# --------------------------------------------------------------
# Transformers
# --------------------------------------------------------------

# INSTALLATION
# conda create -n nlp python=3.10
# conda env remove --name nlp

# conda reinstall:
# conda init --reverse --all
# rm -rf anaconda3

# xcode-select -p
# xcode-select --install (maybe already done)

# conda install -c apple tensorflow-deps
# python -m pip install tensorflow-macos
# python -m pip install tensorflow-metal
# pip install transformers
# pip install tf-keras
# pip install black

import sys
import tensorflow as tf
import tensorflow.keras

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")

# Updated GPU availability check
gpu_available = tf.config.list_physical_devices("GPU")
print("GPU is", "available" if gpu_available else "NOT AVAILABLE")

# pip freeze > requirements.txt

# --------------------------------------------------------------
# OpenAI
# --------------------------------------------------------------

# TODO: Azure Document Intelligence
# TODO: Pydantic

# INSTALLATION
# pip install python-dotenv
# pip isntall OpenAI
# pip install instructor

from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()
# OR
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

stream = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "Say this is a test"}],
    stream=True,
)

for chunk in stream:
    if chunk.choices[0].delta.content is not None:
        print(chunk.choices[0].delta.content, end="")
