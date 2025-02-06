from langchain_core.tools import tool
import requests

@tool
def get_coordinates_by_query(query: str) -> dict|str:
    """Get coordinates by doing a location search"""

    response = requests.get(f"https://geocoding-api.open-meteo.com/v1/search?name={query}")
    data = response.json()

    # check if a location is found
    if "results" in data and len(data["results"]) > 0:
        return data["results"][0]
    
    return f"No location found for {query}"