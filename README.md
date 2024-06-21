# NLP-Knowledge

## Transformers

### Installation

Follow these steps to set up your environment for using transformers:

1. **Create a Conda Environment:**
    ```sh
    conda create -n nlp python=3.10
    ```

2. **Remove Conda Environment (in case something went wrong):**
    ```sh
    conda env remove --name nlp
    ```

3. **Install Xcode Command Line Tools:**
    Check if Xcode command line tools are installed:
    ```sh
    xcode-select -p
    ```
    If not installed, you can install them using:
    ```sh
    xcode-select --install
    ```

4. **Install TensorFlow Dependencies for macOS:**
    ```sh
    conda install -c apple tensorflow-deps
    ```

5. **Install TensorFlow for macOS:**
    ```sh
    python -m pip install tensorflow-macos
    python -m pip install tensorflow-metal
    ```

6. **Install Additional Packages:**
    ```sh
    pip install transformers
    pip install tf-keras
    ```

## OpenAI

### Installation

Follow these steps to set up your environment for using OpenAI:

1. **Install `python-dotenv`:**
    ```sh
    pip install python-dotenv
    ```

2. **Install OpenAI Package:**
    ```sh
    pip install openai
    ```

---

By following the steps above, you will set up your environment for both Transformers and OpenAI development. Ensure you have Conda and Python properly installed on your machine before proceeding with these steps.
