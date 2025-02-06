from langchain_core.tools import tool
import requests

@tool
def get_temperature_from_lat_lon(latitude: float, longitude: float) -> str:
    """Get the current temperature for given coordinates"""

    response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={latitude}2&longitude={longitude}&hourly=temperature_2m")
    data = response.json()

    # check if the temperature is found
    if "hourly" in data and "time" in data["hourly"]:
        return str(data["hourly"]["temperature_2m"][0]) + "°C"

    return "Temperature not found"