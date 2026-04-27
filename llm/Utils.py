import os
from dotenv import load_dotenv

class Utils:
    def __init__(self):
        load_dotenv()

    def get_api_key(self):
        return os.getenv("UNSTRUCTURED_API_KEY")

    def get_url(self):
        return os.getenv("UNSTRUCTURED_API_URL")