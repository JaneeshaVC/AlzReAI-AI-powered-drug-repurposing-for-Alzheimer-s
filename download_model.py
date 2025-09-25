import requests
import os

def download_model():
    if not os.path.exists('model_components.pkl'):
        print("Downloading model...")
        # Example with Google Drive public link
        url = "https://drive.google.com/file/d/1aL5FqzF_SlDChgePrAla8RGYlMt2OJfp/view?usp=sharing"
        response = requests.get(url)
        with open('model_components.pkl', 'wb') as f:
            f.write(response.content)
        print("Model downloaded successfully")

if __name__ == "__main__":
    download_model()