import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date, time, timedelta
import requests
import folium
from folium.plugins import Geocoder
from streamlit_folium import st_folium
import pickle
from geopy.geocoders import Nominatim

# ------------------ APP CONFIG ------------------
st.set_page_config(page_title="Will it Rain on My Parade?", page_icon="🌦️", layout="wide")

# ------------------ MODEL & API FUNCTIONS ------------------

@st.cache_resource
def load_model(model_path):
    """Loads the pre-trained XGBoost model from the specified .pkl file."""
    try:
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except FileNotFoundError:
        st.error(f"Error: The model file '{model_path}' was not found. Make sure it's in the same directory.")
        return None
    except Exception as e:
        st.error(f"Error loading the model: {e}")
        return None

@st.cache_data(ttl="1h")
def get_forecast_data(latitude, longitude, event_date):
    """Fetches both hourly and daily forecast data to be used as input for the model."""
    url = f"https://api.open-meteo.com/v1/forecast?latitude={latitude}&longitude={longitude}&start_date={event_date}&end_date={event_date}&hourly=temperature_2m,relative_humidity_2m,precipitation_probability&daily=temperature_2m_max,temperature_2m_min,windspeed_10m_max"
    try:
        res = requests.get(url)
        res.raise_for_status()
        return res.json()
    except Exception as e:
        st.error(f"Could not fetch forecast data for the model: {e}")
        return None

@st.cache_data(ttl="6h")
def get_historical_daily_rain(latitude, longitude, event_date):
    """Fetches daily rain totals for the same date for the past 5 years using the API."""
    history_data = []
    for i in range(5, 0, -1):
        past_date = event_date - timedelta(days=i*365)
        url = f"https://archive-api.open-meteo.com/v1/archive?latitude={latitude}&longitude={longitude}&start_date={past_date}&end_date={past_date}&daily=precipitation_sum"
        try:
            res = requests.get(url).json()
            history_data.append({'Year': past_date.year, 'Rainfall (mm)': res['daily']['precipitation_sum'][0]})
        except Exception:
            continue
    return pd.DataFrame(history_data)

# ------------------ SESSION STATE INITIALIZATION ------------------
if 'latitude' not in st.session_state:
    st.session_state.latitude = 28.40  # Default location
if 'longitude' not in st.session_state:
    st.session_state.longitude = 77.31
if 'event_date' not in st.session_state:
    st.session_state.event_date = date.today() + timedelta(days=1)
if 'event_time' not in st.session_state:
    st.session_state.event_time = time(18, 0)
if 'event_type' not in st.session_state:
    st.session_state.event_type = "Picnic"
if 'crowd_size' not in st.session_state:
    st.session_state.crowd_size = 500

# ------------------ UI LAYOUT ------------------
st.title("🌦️ Will it Rain on My Parade?")
st.subheader("Plan event in a smarter way on weather conditions.")

# Load the trained model
model = load_model('daily_rain_classifier.pkl')

with st.sidebar:
    st.header("🔎 Navigation")
    section = st.radio("Go to:", ["Event Input", "Forecast Dashboard", "History", "Suggestions"])

if section == "Event Input":
    st.header("🎭 Plan Your Event")
    
    # Location Search Bar
    location_name = st.text_input("Search for a location:", "Faridabad, India")
    geolocator = Nominatim(user_agent="event_weather_app")
    try:
        location = geolocator.geocode(location_name)
        if location:
            st.session_state.latitude = location.latitude
            st.session_state.longitude = location.longitude
            st.success(f"Found '{location.address}'. Map centered.")
        else:
            st.warning("Location not found. Please try another search.")
    except Exception as e:
        st.error(f"Geocoding service error: {e}")

    st.info("Click on the map to fine-tune the precise spot.")
    m = folium.Map(location=[st.session_state.latitude, st.session_state.longitude], zoom_start=10)
    map_data = st_folium(m, width=1200, height=400)
    
    if map_data and map_data['last_clicked']:
        st.session_state.latitude = map_data['last_clicked']['lat']
        st.session_state.longitude = map_data['last_clicked']['lng']
    
    st.success(f"📍 Location set: Latitude={st.session_state.latitude:.4f}, Longitude={st.session_state.longitude:.4f}")
    st.markdown("---")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.session_state.event_date = st.date_input("📅 Event Date", st.session_state.event_date)
    with col2:
        st.session_state.event_time = st.time_input("⏰ Event Time", st.session_state.event_time)
    with col3:
        st.session_state.event_type = st.selectbox("🎉 Event Type:", ["Picnic", "Concert", "Wedding", "Sports", "Other"], index=["Picnic", "Concert", "Wedding", "Sports", "Other"].index(st.session_state.event_type))
    with col4:
        st.session_state.crowd_size = st.slider("👥 Expected Crowd:", 50, 10000, st.session_state.crowd_size)

# Fetch data needed for the other sections
forecast_data = get_forecast_data(st.session_state.latitude, st.session_state.longitude, st.session_state.event_date)

if section == "Forecast Dashboard":
    st.header(f"Forecast for {st.session_state.event_date}")
    if model and forecast_data and 'daily' in forecast_data:
        daily_data = forecast_data['daily']
        input_df = pd.DataFrame({
            'temperature_2m_max': [daily_data['temperature_2m_max'][0]],
            'temperature_2m_min': [daily_data['temperature_2m_min'][0]],
            'windspeed_10m_max': [daily_data['windspeed_10m_max'][0]]
        })
        
        prediction_proba = model.predict_proba(input_df)[0][1]

        col1, col2, col3 = st.columns(3)
        col1.metric("Rain Probability", f"{prediction_proba*100:.1f}%")
        col2.metric("Max Temperature", f"{daily_data['temperature_2m_max'][0]}°C")
        col3.metric("Max Wind Speed", f"{daily_data['windspeed_10m_max'][0]} km/h")

        st.markdown("---")
        st.subheader("Hourly Forecast")
        hourly_df = pd.DataFrame(forecast_data['hourly'])
        hourly_df['time'] = pd.to_datetime(hourly_df['time'])
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(hourly_df["time"], hourly_df["precipitation_probability"], marker="o", color="blue", label="API Rain Probability (%)")
        ax.axvline(x=pd.to_datetime(f"{st.session_state.event_date} {st.session_state.event_time}"), color='r', linestyle='--', label='Event Time')
        ax.set_title("Hourly Rain Forecast (from API)"); ax.set_xlabel("Hour of Day"); ax.set_ylabel("Rain Probability (%)")
        ax.grid(True); ax.legend()
        st.pyplot(fig)
    else:
        st.warning("Could not load model or forecast data for the dashboard.")

if section == "History":
    st.header(f"📊Rainfall History for {st.session_state.event_date.strftime('%B %d')}")
    history_df = get_historical_daily_rain(st.session_state.latitude, st.session_state.longitude, st.session_state.event_date)
    if history_df is not None and not history_df.empty:
        fig2, ax2 = plt.subplots()
        ax2.bar(history_df["Year"].astype(str), history_df["Rainfall (mm)"], color="skyblue")
        ax2.set_title(f"Total Rainfall on {st.session_state.event_date.strftime('%b %d')} (Past 5 Years)")
        ax2.set_xlabel("Year"); ax2.set_ylabel("Total Rainfall (mm)")
        st.pyplot(fig2)
    else:
        st.warning("Could not load historical data from the API.")

if section == "Suggestions":
    st.header("Suggestions for Event")
    if model and forecast_data and 'daily' in forecast_data:
        daily_data = forecast_data['daily']
        input_df = pd.DataFrame({
            'temperature_2m_max': [daily_data['temperature_2m_max'][0]],
            'temperature_2m_min': [daily_data['temperature_2m_min'][0]],
            'windspeed_10m_max': [daily_data['windspeed_10m_max'][0]]
        })

        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0][1]

        col1, col2 = st.columns(2)
        col1.metric("Model Prediction: Will it Rain?", "Yes" if prediction == 1 else "No")
        col2.metric("Chances of Rain", f"{prediction_proba*100:.1f}%")

        if prediction == 1 and prediction_proba > 0.6:
            risk_level = "High Risk of Rain 🌧️"
            suggestion = f"⚠️ The model predicts rain with {prediction_proba*100:.0f}% confidence. Activating an indoor backup plan is strongly recommended."
            st.error(risk_level); st.warning(suggestion)
        elif prediction == 1:
            risk_level = "Moderate Risk of Showers 🌦️"
            suggestion = f"The model predicts rain with {prediction_proba*100:.0f}% confidence. Having tents or covered areas on standby is a wise precaution."
            st.warning(risk_level); st.info(suggestion)
        else:
            risk_level = "Low Risk of Rain ☀️"
            suggestion = f"✅ The model is {100-prediction_proba*100:.0f}% confident it will NOT rain. Conditions look favorable."
            st.success(risk_level); st.info(suggestion)
    else:
        st.error("The model (`daily_rain_classifier.pkl`) or forecast data is unavailable.")