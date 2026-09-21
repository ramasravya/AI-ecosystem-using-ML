import requests

API_KEY = "3e90df8ec976fd799d8f9d9319e50dd5"

def get_weather(city):

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

    response = requests.get(url)

    if response.status_code != 200:
        return None

    data = response.json()

    return {
        "city": data["name"],
        "temperature": data["main"]["temp"],
        "humidity": data["main"]["humidity"],
        "pressure": data["main"]["pressure"],
        "wind": data["wind"]["speed"],
        "weather": data["weather"][0]["description"],
        "lat": data["coord"]["lat"],
        "lon": data["coord"]["lon"]
    }
if __name__ == "__main__":
    city = input("Enter City Name: ")

    result = get_weather(city)

    if result:
        print("\n===== WEATHER DATA =====")
        print("City:", result["city"])
        print("Temperature:", result["temperature"], "°C")
        print("Humidity:", result["humidity"], "%")
        print("Pressure:", result["pressure"], "hPa")
        print("Wind Speed:", result["wind"], "m/s")
        print("Weather:", result["weather"])
        print("Latitude:", result["lat"])
        print("Longitude:", result["lon"])
    else:
        print("City not found!")