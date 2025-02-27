from .base_auth import BaseAuth

class KeyAuth(BaseAuth):

    def __init__(self, api_key: str):
        self.api_key = api_key

    def authenticate(self, api_key: str):
        return api_key == self.api_key