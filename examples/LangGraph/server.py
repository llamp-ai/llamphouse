from llamphouse.core import LLAMPHouse
from dotenv import load_dotenv

# Import the assistant
from assistants.temperature import TemperatureAssistant

# Load the environment variables from the .env file
load_dotenv(override=True) 

def main():
    # Create an instance of the assistant
    temperature_assistant = TemperatureAssistant("Temperature Assistant")

    # Create a new LLAMPHouse instance with the assistant
    llamphouse = LLAMPHouse(assistants=[temperature_assistant], worker_type="async", api_key="secret_key")
    llamphouse.ignite(host="127.0.0.1", port=8000)

if __name__ == "__main__":
    main()