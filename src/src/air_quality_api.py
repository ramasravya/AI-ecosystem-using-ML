import requests

API_KEY = "3e90df8ec976fd799d8f9d9319e50dd5"

city = input("Enter City Name: ")

# Step 1: Get weather details (includes latitude & longitude)
weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

weather_response = requests.get(weather_url)
weather_data = weather_response.json()

if weather_response.status_code == 200:

    lat = weather_data["coord"]["lat"]
    lon = weather_data["coord"]["lon"]

    print("\nLocation:", weather_data["name"])
    print("Latitude :", lat)
    print("Longitude:", lon)

    # Step 2: Get Air Quality Data
    air_url = f"https://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}"

    air_response = requests.get(air_url)
    air_data = air_response.json()

    aqi = air_data["list"][0]["main"]["aqi"]
    components = air_data["list"][0]["components"]

    print("\n========== AIR QUALITY ==========")

    if aqi == 1:
        quality = "Good 😊"
    elif aqi == 2:
        quality = "Fair 🙂"
    elif aqi == 3:
        quality = "Moderate 😐"
    elif aqi == 4:
        quality = "Poor 😷"
    else:
        quality = "Very Poor ☠"

    print("AQI:", quality)
    print("CO   :", components["co"])
    print("NO   :", components["no"])
    print("NO2  :", components["no2"])
    print("O3   :", components["o3"])
    print("SO2  :", components["so2"])
    print("PM2.5:", components["pm2_5"])
    print("PM10 :", components["pm10"])
    print("NH3  :", components["nh3"])

else:
    print("City not found!")