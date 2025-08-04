import openmeteo_requests
import requests_cache
import pandas as pd
from retry_requests import retry
from datetime import datetime, timedelta

def get_temperature_data(latitude, longitude, forecast_days=1, timezone="auto"):
    """
    Get temperature data from Open-Meteo API for given coordinates using the openmeteo_requests client.

    Parameters:
    -----------
    latitude : float
        Latitude coordinate (e.g., 35.1856 for Cyprus)
    longitude : float
        Longitude coordinate (e.g., 33.3823 for Cyprus)
    forecast_days : int, optional
        Number of forecast days (default 1)
    timezone : str, optional
        Timezone string (default "auto")

    Returns:
    --------
    pd.DataFrame : DataFrame with time and temperature columns
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m",
        "timezone": timezone,
        "forecast_days": forecast_days
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly_temperature_2m
        }
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        return hourly_dataframe
    except Exception as e:
        print(f"Failed to retrieve temperature data: {e}")
        return pd.DataFrame()

def get_historic_temperature_data(latitude, longitude, date=None, start_date=None, end_date=None, timezone="auto"):
    """
    Get historic temperature data for a specific day or date range from Open-Meteo API.

    Parameters:
    -----------
    latitude : float
        Latitude coordinate (e.g., 35.1856 for Cyprus)
    longitude : float
        Longitude coordinate (e.g., 33.3823 for Cyprus)
    date : str or datetime, optional
        Date string in 'YYYY-MM-DD' format or datetime object for the day to retrieve historic data.
        If provided, start_date and end_date are ignored.
    start_date : str or datetime, optional
        Start date string in 'YYYY-MM-DD' format or datetime object for the range.
    end_date : str or datetime, optional
        End date string in 'YYYY-MM-DD' format or datetime object for the range (exclusive).
    timezone : str, optional
        Timezone string (default "auto")

    Returns:
    --------
    pd.DataFrame : DataFrame with time and temperature columns for the specified day or range
    """
    # Setup the Open-Meteo API client with cache and retry on error
    cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
    retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
    openmeteo = openmeteo_requests.Client(session=retry_session)

    url = "https://archive-api.open-meteo.com/v1/archive"

    # Determine start_date and end_date
    if date is not None:
        # Single day mode
        if isinstance(date, str):
            date_obj = datetime.strptime(date, "%Y-%m-%d")
        elif isinstance(date, datetime):
            date_obj = date
        else:
            raise ValueError("date must be a string in 'YYYY-MM-DD' format or a datetime object.")
        start_date_str = date_obj.strftime("%Y-%m-%d")
        end_date_str = (date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
    elif start_date is not None and end_date is not None:
        # Range mode
        if isinstance(start_date, str):
            start_date_obj = datetime.strptime(start_date, "%Y-%m-%d")
        elif isinstance(start_date, datetime):
            start_date_obj = start_date
        else:
            raise ValueError("start_date must be a string in 'YYYY-MM-DD' format or a datetime object.")
        if isinstance(end_date, str):
            end_date_obj = datetime.strptime(end_date, "%Y-%m-%d")
        elif isinstance(end_date, datetime):
            end_date_obj = end_date
        else:
            raise ValueError("end_date must be a string in 'YYYY-MM-DD' format or a datetime object.")
        start_date_str = start_date_obj.strftime("%Y-%m-%d")
        end_date_str = end_date_obj.strftime("%Y-%m-%d")
    else:
        raise ValueError("You must provide either 'date' or both 'start_date' and 'end_date'.")

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date_str,
        "end_date": end_date_str,
        "hourly": "temperature_2m",
        "timezone": timezone
    }
    try:
        responses = openmeteo.weather_api(url, params=params)
        response = responses[0]
        hourly = response.Hourly()
        hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
        hourly_data = {
            "date": pd.date_range(
                start=pd.to_datetime(hourly.Time() + response.UtcOffsetSeconds(), unit="s"),
                end=pd.to_datetime(hourly.TimeEnd() + response.UtcOffsetSeconds(), unit="s"),
                freq=pd.Timedelta(seconds=hourly.Interval()),
                inclusive="left"
            ),
            "temperature_2m": hourly_temperature_2m
        }
        hourly_dataframe = pd.DataFrame(data=hourly_data)
        return hourly_dataframe
    except Exception as e:
        print(f"Failed to retrieve historic temperature data: {e}")
        return pd.DataFrame()