# INSTALLATION
# conda create -n nlp python=3.10
# conda env remove --name nlp

# conda activate
# conda init --reverse --all
# rm -rf anaconda3

# xcode-select -p
# xcode-select --install

# conda install -c apple tensorflow-deps
# python -m pip install tensorflow-macos
# python -m pip install tensorflow-metal
# pip install black
# pip install transformers
# pip install tf-keras

# What version do you have?
import sys
import tensorflow.keras
import tensorflow as tf

print(f"Tensor Flow Version: {tf.__version__}")
print(f"Keras Version: {tensorflow.keras.__version__}")
print()
print(f"Python {sys.version}")
print("GPU is", "available" if tf.test.is_gpu_available() else "NOT AVAILABLE")

# pip freeze > requirements.txt
