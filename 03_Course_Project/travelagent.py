import base64
import streamlit as st
import json
import os
import requests
import pandas as pd
from serpapi import GoogleSearch
from agno.agent import Agent
from agno.tools.serpapi import SerpApiTools
from agno.models.google import Gemini
from datetime import datetime, timedelta
from dateutil import tz



def get_svg_base64(file_path):
    """Read SVG file and encode as base64 data URI"""
    try:
        with open(file_path, "rb") as file:
            svg_content = file.read()
        base64_encoded = base64.b64encode(svg_content).decode('utf-8')
        return f"data:image/svg+xml;base64,{base64_encoded}"
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"Error reading SVG file: {e}")
        return None
    

GOOGLE_MODEL= "gemini-2.0-flash"

# Load OpenFlights airports dataset
@st.cache_resource
def load_airports_data():
    url = "https://raw.githubusercontent.com/jpatokal/openflights/master/data/airports.dat"
    columns = ["id", "name", "city", "country", "iata", "icao", "lat", "lon", "altitude", "timezone", "dst", "tz_database", "type", "source"]
    df = pd.read_csv(url, header=None, names=columns, na_values="\\N")
    df = df[df["iata"].notna() & df["lat"].notna() & df["lon"].notna()]
    return df.set_index("iata")[["name", "lat", "lon", "city", "country"]].to_dict("index")

# Set up Streamlit UI with a travel-friendly theme
st.set_page_config(page_title="🌍 AI Travel Planner", layout="wide")
st.markdown(
    """
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            color: #ff5733;
        }
        .subtitle {
            text-align: center;
            font-size: 20px;
            color: #555;
        }
        .stSlider > div {
            background-color: #f9f9f9;
            padding: 10px;
            border-radius: 10px;
        }
        .weather-card {
            border: 2px solid #ddd;
            border-radius: 10px;
            padding: 15px;
            text-align: center;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
            background-color: #e6f3fa;
            margin-bottom: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and subtitle
st.markdown('<h1 class="title">✈️ AI-Powered Travel Planner</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Plan your dream trip with AI! Get personalized recommendations for flights, hotels, activities, and weather forecasts.</p>', unsafe_allow_html=True)

# User Inputs Section
st.markdown("### 🌍 Where are you headed?")
source = st.text_input("🛫 Departure City (IATA Code):", "SDF")  # Example: Louisville for Kentucky
destination = st.text_input("🛬 Destination (IATA Code):", "JFK")  # Example: John F. Kennedy for New York

st.markdown("### 📅 Plan Your Adventure")
num_days = st.slider("🕒 Trip Duration (days):", 1, 14, 5)
travel_theme = st.selectbox(
    "🎭 Select Your Travel Theme:",
    ["💑 Couple Getaway", "👨‍👩‍👧‍👦 Family Vacation", "🏔️ Adventure Trip", "🧳 Solo Exploration"]
)

# Divider for aesthetics
st.markdown("---")

st.markdown(
    f"""
    <div style="
        text-align: center; 
        padding: 15px; 
        background-color: #ffecd1; 
        border-radius: 10px; 
        margin-top: 20px;
    ">
        <h3>🌟 Your {travel_theme} to {destination} is about to begin! 🌟</h3>
        <p>Let's find the best flights, stays, experiences, and weather forecasts for your unforgettable journey.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

activity_preferences = st.text_area(
    "🌍 What activities do you enjoy? (e.g., relaxing on the beach, exploring historical sites, nightlife, adventure)",
    "Relaxing on the beach, exploring historical sites"
)

departure_date = st.date_input("Departure Date")
return_date = st.date_input("Return Date")

# Sidebar Setup
st.sidebar.title("🌎 Travel Assistant")
st.sidebar.subheader("Personalize Your Trip")

# Travel Preferences
budget = st.sidebar.radio("💰 Budget Preference:", ["Economy", "Standard", "Luxury"])
flight_class = st.sidebar.radio("✈️ Flight Class:", ["Economy", "Business", "First Class"])
hotel_rating = st.sidebar.selectbox("🏨 Preferred Hotel Rating:", ["Any", "3⭐", "4⭐", "5⭐"])

# Packing Checklist
st.sidebar.subheader("🎒 Packing Checklist")
packing_list = {
    "👕 Clothes": True,
    "🩴 Comfortable Footwear": True,
    "🕶️ Sunglasses & Sunscreen": False,
    "📖 Travel Guidebook": False,
    "💊 Medications & First-Aid": True
}
for item, checked in packing_list.items():
    st.sidebar.checkbox(item, value=checked)

# Travel Essentials
st.sidebar.subheader("🛂 Travel Essentials")
visa_required = st.sidebar.checkbox("🛃 Check Visa Requirements")
travel_insurance = st.sidebar.checkbox("🛡️ Get Travel Insurance")
currency_converter = st.sidebar.checkbox("💱 Currency Exchange Rates")

# API Keys (only SerpApi and Google API needed)
SERPAPI_KEY = "YOUR_SERPAPI_KEY"
GOOGLE_API_KEY = "YOUR_GOOGLE_API_KEY"
# SERPAPI_KEY = ""
# GOOGLE_API_KEY = ""
# Validate API keys
if not SERPAPI_KEY or not GOOGLE_API_KEY:
    st.error("SerpApi or Google API key is missing. Please configure them in environment variables or Streamlit secrets.")
    st.stop()

# Load airports data
IATA_COORDINATES = load_airports_data()

# Weather code to description and icon mapping (adapted from provided JSON)
WEATHER_CODE_MAP = {
    0: {"description": "Clear Sky", "icons": {"day": "assets/clear.svg", "night": "assets/clear-night.svg"}},
    1: {"description": "Mainly Clear", "icons": {"day": "assets/clear.svg", "night": "assets/clear-night.svg"}},
    2: {"description": "Partly Cloudy", "icons": {"day": "assets/partly-cloudy.svg", "night": "assets/partly-cloudy-night.svg"}},
    3: {"description": "Overcast", "icons": {"day": "assets/overcast.svg", "night": "assets/overcast.svg"}},
    45: {"description": "Fog", "icons": {"day": "assets/fog.svg", "night": "assets/fog-night.svg"}},
    48: {"description": "Rime Fog", "icons": {"day": "assets/rime-fog.svg", "night": "assets/rime-fog.svg"}},
    51: {"description": "Light Drizzle", "icons": {"day": "assets/light-drizzle.svg", "night": "assets/light-drizzle.svg"}},
    53: {"description": "Moderate Drizzle", "icons": {"day": "assets/drizzle.svg", "night": "assets/drizzle.svg"}},
    55: {"description": "Heavy Drizzle", "icons": {"day": "assets/heavy-drizzle.svg", "night": "assets/heavy-drizzle.svg"}},
    56: {"description": "Light Freezing Drizzle", "icons": {"day": "assets/drizzle.svg", "night": "assets/drizzle.svg"}},
    57: {"description": "Dense Freezing Drizzle", "icons": {"day": "assets/heavy-drizzle.svg", "night": "assets/heavy-drizzle.svg"}},
    61: {"description": "Slight Rain", "icons": {"day": "assets/slight-rain.svg", "night": "assets/slight-rain-night.svg"}},
    63: {"description": "Moderate Rain", "icons": {"day": "assets/rain.svg", "night": "assets/rain.svg"}},
    65: {"description": "Heavy Rain", "icons": {"day": "assets/heavy-rain.svg", "night": "assets/heavy-rain.svg"}},
    66: {"description": "Light Freezing Rain", "icons": {"day": "assets/rain.svg", "night": "assets/rain.svg"}},
    67: {"description": "Heavy Freezing Rain", "icons": {"day": "assets/heavy-rain.svg", "night": "assets/heavy-rain.svg"}},
    71: {"description": "Slight Snowfall", "icons": {"day": "assets/light-snow.svg", "night": "assets/light-snow-night.svg"}},
    73: {"description": "Moderate Snowfall", "icons": {"day": "assets/snow.svg", "night": "assets/snow.svg"}},
    75: {"description": "Heavy Snowfall", "icons": {"day": "assets/heavy-snow.svg", "night": "assets/heavy-snow.svg"}},
    77: {"description": "Snow Grains", "icons": {"day": "assets/snow-grains.svg", "night": "assets/snow-grains.svg"}},
    80: {"description": "Slight Rain Showers", "icons": {"day": "assets/slight-rain-showers.svg", "night": "assets/slight-rain-showers-night.svg"}},
    81: {"description": "Moderate Rain Showers", "icons": {"day": "assets/rain-showers.svg", "night": "assets/rain-showers.svg"}},
    82: {"description": "Violent Rain Showers", "icons": {"day": "assets/heavy-rain-showers.svg", "night": "assets/heavy-rain-showers.svg"}},
    85: {"description": "Light Snow Showers", "icons": {"day": "assets/light-snow-showers.svg", "night": "assets/light-snow-showers.svg"}},
    86: {"description": "Heavy Snow Showers", "icons": {"day": "assets/heavy-snow-showers.svg", "night": "assets/heavy-snow-showers.svg"}},
    95: {"description": "Thunderstorm", "icons": {"day": "assets/thunderstorm.svg", "night": "assets/thunderstorm.svg"}},
    96: {"description": "Slight Hailstorm", "icons": {"day": "assets/hail.svg", "night": "assets/hail.svg"}},
    99: {"description": "Heavy Hailstorm", "icons": {"day": "assets/heavy-hail.svg", "night": "assets/heavy-hail.svg"}}
}

# Function to determine if it's day or night based on local time
def is_daytime(date_str, timezone_str):
    try:
        # Parse the forecast date
        forecast_date = datetime.strptime(date_str, "%Y-%m-%d")
        # Assume midday (12 PM) for simplicity, as Open-Meteo provides daily data
        forecast_datetime = forecast_date.replace(hour=12, minute=0, second=0)
        # Convert to destination timezone
        local_tz = tz.gettz(timezone_str)
        local_time = forecast_datetime.replace(tzinfo=tz.UTC).astimezone(local_tz)
        # Daytime: 6 AM to 6 PM
        return 6 <= local_time.hour < 18
    except Exception:
        # Default to day if timezone parsing fails
        return True

# Function to fetch weather data from Open-Meteo
@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_weather(lat, lon, start_date, end_date):
    try:
        # Validate date range
        today = datetime.now().date()
        max_forecast_date = today + timedelta(days=7)
        start_date_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        if start_date_dt > max_forecast_date:
            st.warning("⚠️ Weather forecasts are not available for dates beyond 7 days from today.")
            return []

        # Construct Open-Meteo API URL
        url = (
            f"https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,weathercode"
            f"&timezone=auto&start_date={start_date}&end_date={end_date}"
        )
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()

        # Get timezone from response
        timezone_str = data.get("timezone", "UTC")

        # Filter daily forecasts for the travel period
        daily_forecasts = data.get("daily", {})
        weather_summary = []
        for i, date in enumerate(daily_forecasts.get("time", [])):
            weather_code = daily_forecasts.get("weathercode", [])[i]
            weather_info = WEATHER_CODE_MAP.get(weather_code, {
                "description": f"Unknown (code: {weather_code})",
                "icons": {"day": "assets/unknown.svg", "night": "assets/unknown.svg"}
            })
            # Determine day or night icon
            icon_key = "day" if is_daytime(date, timezone_str) else "night"
            weather_summary.append({
                "date": datetime.strptime(date, "%Y-%m-%d").strftime("%b-%d, %Y"),
                "temp_min": daily_forecasts.get("temperature_2m_min", [])[i],
                "temp_max": daily_forecasts.get("temperature_2m_max", [])[i],
                "description": weather_info["description"],
                "icon": weather_info["icons"][icon_key]
            })
        return weather_summary
    except Exception as e:
        st.warning(f"Failed to fetch weather data: {str(e)}")
        return []

# Function to fetch flight data
def fetch_flights(source, destination, departure_date, return_date):
    params = {
        "engine": "google_flights",
        "departure_id": source,
        "arrival_id": destination,
        "outbound_date": str(departure_date),
        "return_date": str(return_date),
        "currency": "INR",
        "hl": "en",
        "api_key": SERPAPI_KEY
    }
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        return results
    except Exception as e:
        st.warning(f"Failed to fetch flight data: {str(e)}")
        return {}

# Function to extract top 3 cheapest flights
def extract_cheapest_flights(flight_data):
    best_flights = flight_data.get("best_flights", [])
    sorted_flights = sorted(best_flights, key=lambda x: x.get("price", float("inf")))[:3]
    return sorted_flights

# Function to format datetime
def format_datetime(iso_string):
    try:
        dt = datetime.strptime(iso_string, "%Y-%m-%d %H:%M")
        return dt.strftime("%b-%d, %Y | %I:%M %p")
    except:
        return "N/A"

# AI Agents
researcher = Agent(
    name="Researcher",
    instructions=[
        "Identify the travel destination specified by the user.",
        "Gather detailed information on the destination, including climate, culture, and safety tips.",
        "Find popular attractions, landmarks, and must-visit places.",
        "Search for activities that match the user’s interests and travel style.",
        "Incorporate weather forecasts (temperature and conditions) to suggest indoor or outdoor activities.",
        "Prioritize information from reliable sources and official travel guides.",
        "Provide well-structured summaries with key insights and recommendations."
    ],
    model=Gemini(id=GOOGLE_MODEL),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

planner = Agent(
    name="Planner",
    instructions=[
        "Gather details about the user's travel preferences and budget.",
        "Create a detailed itinerary with scheduled activities and estimated costs.",
        "Ensure the itinerary includes transportation options and travel time estimates.",
        "Incorporate weather forecasts (temperature and conditions) to adjust activity recommendations (e.g., indoor activities for rainy days).",
        "Optimize the schedule for convenience and enjoyment.",
        "Present the itinerary in a structured format."
    ],
    model=Gemini(id=GOOGLE_MODEL),
    add_datetime_to_instructions=True,
)

hotel_restaurant_finder = Agent(
    name="Hotel & Restaurant Finder",
    instructions=[
        "Identify key locations in the user's travel itinerary.",
        "Search for highly rated hotels near those locations.",
        "Search for top-rated restaurants based on cuisine preferences and proximity.",
        "Prioritize results based on user preferences, ratings, and availability.",
        "Provide direct booking links or reservation options where possible."
    ],
    model=Gemini(id=GOOGLE_MODEL),
    tools=[SerpApiTools(api_key=SERPAPI_KEY)],
    add_datetime_to_instructions=True,
)

# Generate Travel Plan
if st.button("🚀 Generate Travel Plan"):
    # Validate inputs
    if source not in IATA_COORDINATES or destination not in IATA_COORDINATES:
        st.error(f"Invalid IATA code: {source} or {destination}. Please use valid IATA codes (e.g., BOM, GOI, DEL, JFK, LHR).")
        st.stop()
    if departure_date >= return_date:
        st.error("Return date must be after departure date.")
        st.stop()

    # Fetch weather data
    with st.spinner("⛅ Fetching weather forecast..."):
        dest_coords = IATA_COORDINATES.get(destination, {})
        # Ensure weather forecast doesn't exceed 7 days from today
        max_forecast_date = (datetime.now().date() + timedelta(days=7))
        effective_end_date = min(return_date, max_forecast_date)
        weather_data = fetch_weather(
            dest_coords.get("lat"), 
            dest_coords.get("lon"), 
            departure_date.strftime("%Y-%m-%d"), 
            effective_end_date.strftime("%Y-%m-%d")
        )

    # Fetch flight data
    with st.spinner("✈️ Fetching best flight options..."):
        flight_data = fetch_flights(source, destination, departure_date, return_date)
        cheapest_flights = extract_cheapest_flights(flight_data)
        print("commnented!!")

    # AI Processing
    with st.spinner("🔍 Researching best attractions & activities..."):
        weather_summary = "\n".join([f"{day['date']}: {day['description']} ({day['temp_min']}°C to {day['temp_max']}°C)" for day in weather_data])
        research_prompt = (
            f"Research the best attractions and activities in {destination} for a {num_days}-day {travel_theme.lower()} trip. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. "
            f"Hotel Rating: {hotel_rating}. Visa Requirement: {visa_required}. Travel Insurance: {travel_insurance}. "
            f"Weather Forecast: {weather_summary}."
        )
        research_results = researcher.run(research_prompt, stream=False)
        # research_results=""


    with st.spinner("🏨 Searching for hotels & restaurants..."):
        hotel_restaurant_prompt = (
            f"Find the best hotels and restaurants near popular attractions in {destination} for a {travel_theme.lower()} trip. "
            f"Budget: {budget}. Hotel Rating: {hotel_rating}. Preferred activities: {activity_preferences}."
        )
        hotel_restaurant_results = hotel_restaurant_finder.run(hotel_restaurant_prompt, stream=False)
        # hotel_restaurant_results=""


    with st.spinner("🗺️ Creating your personalized itinerary..."):
        planning_prompt = (
            f"Based on the following data, create a {num_days}-day itinerary for a {travel_theme.lower()} trip to {destination}. "
            f"The traveler enjoys: {activity_preferences}. Budget: {budget}. Flight Class: {flight_class}. Hotel Rating: {hotel_rating}. "
            f"Visa Requirement: {visa_required}. Travel Insurance: {travel_insurance}. Research: {research_results.content}. "
            f"Flights: {json.dumps(cheapest_flights)}. Hotels & Restaurants: {hotel_restaurant_results.content}. "
            f"Weather Forecast: {weather_summary}."
        )
        itinerary = planner.run(planning_prompt, stream=False)
        # itinerary = ""

# <img src="{day['icon']}" width="50" alt="Weather Icon" />

    # Display Results
    st.subheader("⛅ Weather Forecast")
    if weather_data:
        if effective_end_date < return_date:
            st.warning("⚠️ Weather forecast is available for up to 7 days from today. For longer trips, check closer to your travel date.")
        cols = st.columns(min(len(weather_data), 3))
        for idx, day in enumerate(weather_data):
            # day['icon'] = "thunderstorm.svg"
            if idx < 3:  # Limit to 3 columns for layout
                with cols[idx]:
                    st.markdown(
                        f"""
                        <div class="weather-card">
                            <h3>{day['date']}</h3>
                            <img src="{get_svg_base64(day['icon'])}" width="50" alt="Weather Icon" />
                            <p><strong>{day['description']}</strong></p>
                            <p>Temp: {day['temp_min']}°C to {day['temp_max']}°C</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.warning("⚠️ No weather data available.")

    st.subheader("✈️ Cheapest Flight Options")
    if cheapest_flights:
        cols = st.columns(len(cheapest_flights))
        for idx, flight in enumerate(cheapest_flights):
            with cols[idx]:
                airline_logo = flight.get("airline_logo", "")
                airline_name = flight.get("airline", "Unknown Airline")
                price = flight.get("price", "Not Available")
                total_duration = flight.get("total_duration", "N/A")
                
                flights_info = flight.get("flights", [{}])
                departure = flights_info[0].get("departure_airport", {})
                arrival = flights_info[-1].get("arrival_airport", {})
                airline_name = flights_info[0].get("airline", "Unknown Airline")
                
                departure_time = format_datetime(departure.get("time", "N/A"))
                arrival_time = format_datetime(arrival.get("time", "N/A"))
                
                departure_token = flight.get("departure_token", "")
                booking_options = flight.get("booking_token", "") if departure_token else ""
                booking_link = f"https://www.google.com/travel/flights?tfs={booking_options}" if booking_options else "#"

                st.markdown(
                    f"""
                    <div style="
                        border: 2px solid #ddd; 
                        border-radius: 10px; 
                        padding: 15px; 
                        text-align: center;
                        box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.1);
                        background-color: #f9f9f9;
                        margin-bottom: 20px;
                    ">
                        <img src="{airline_logo}" width="100" alt="Flight Logo" />
                        <h3 style="margin: 10px 0;">{airline_name}</h3>
                        <p><strong>Departure:</strong> {departure_time}</p>
                        <p><strong>Arrival:</strong> {arrival_time}</p>
                        <p><strong>Duration:</strong> {total_duration} min</p>
                        <h2 style="color: #008000;">💰 {price}</h2>
                        <a href="{booking_link}" target="_blank" style="
                            display: inline-block;
                            padding: 10px 20px;
                            font-size: 16px;
                            font-weight: bold;
                            color: #fff;
                            background-color: #007bff;
                            text-decoration: none;
                            border-radius: 5px;
                            margin-top: 10px;
                        ">🔗 Book Now</a>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    else:
        st.warning("⚠️ No flight data available.")

    st.subheader("🏨 Hotels & Restaurants")
    st.write(hotel_restaurant_results.content)

    st.subheader("🗺️ Your Personalized Itinerary")
    st.write(itinerary.content)

    st.success("✅ Travel plan generated successfully!")