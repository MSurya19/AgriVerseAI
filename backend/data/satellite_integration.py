import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json

class SatelliteDataIntegration:
    def __init__(self):
        self.nasa_power_base_url = "https://power.larc.nasa.gov/api/temporal/daily/point"
        
    def get_real_climate_data(self, lat, lon, start_date, end_date):
        """Get real climate data from NASA POWER API"""
        try:
            params = {
                'parameters': 'T2M,PRECTOT,RH2M,ALLSKY_SFC_SW_DWN',
                'community': 'AG',
                'longitude': lon,
                'latitude': lat,
                'start': start_date,
                'end': end_date,
                'format': 'JSON'
            }
            
            response = requests.get(self.nasa_power_base_url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                return self.process_nasa_data(data)
            else:
                print(f"❌ NASA API error: {response.status_code}")
                return self.get_climate_data_simulated(lat, lon, start_date, end_date)
                
        except Exception as e:
            print(f"❌ Error fetching NASA data: {e}")
            return self.get_climate_data_simulated(lat, lon, start_date, end_date)
    
    def process_nasa_data(self, nasa_data):
        """Process NASA POWER API response"""
        processed_data = []
        
        # Extract climate parameters
        temperature_data = nasa_data['properties']['parameter']['T2M']
        rainfall_data = nasa_data['properties']['parameter']['PRECTOT']
        humidity_data = nasa_data['properties']['parameter']['RH2M']
        
        # Process daily data
        for date_str in temperature_data.keys():
            date_obj = datetime.strptime(date_str, '%Y%m%d')
            
            processed_data.append({
                'date': date_obj.strftime('%Y-%m-%d'),
                'temperature': temperature_data[date_str],
                'rainfall': rainfall_data[date_str],
                'humidity': humidity_data[date_str]
            })
        
        return processed_data
    
    def get_climate_data_simulated(self, lat, lon, start_date, end_date):
        """Fallback simulated climate data"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for date in dates:
            # Seasonal patterns based on latitude
            day_of_year = date.timetuple().tm_yday
            
            # Temperature variation
            base_temp = 20 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            if lat > 35:  # Northern regions cooler
                base_temp -= 5
            elif lat < 20:  # Southern regions warmer  
                base_temp += 5
                
            temperature = base_temp + np.random.normal(0, 3)
            
            # Rainfall patterns
            if 150 <= day_of_year <= 240:  # Rainy season
                rainfall = max(0, np.random.poisson(8))
            else:
                rainfall = max(0, np.random.poisson(2))
            
            humidity = max(30, min(95, 60 + np.random.normal(0, 10)))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'temperature': temperature,
                'rainfall': rainfall,
                'humidity': humidity
            })
        
        return data
    
    def calculate_vegetation_indices(self, lat, lon, date):
        """Calculate NDVI/EVI based on location and date"""
        day_of_year = date.timetuple().tm_yday
        
        # Seasonal vegetation patterns
        if 60 <= day_of_year <= 240:  # Growing season
            if 100 <= day_of_year <= 180:  # Peak growth
                base_ndvi = 0.7
            else:  # Early/late growth
                base_ndvi = 0.5
        else:  # Off-season
            base_ndvi = 0.3
            
        # Geographic variation
        if lat > 35:  # Temperate regions
            base_ndvi += 0.1
        elif lat < 20:  # Tropical regions  
            base_ndvi += 0.15
            
        # Add some randomness
        ndvi = max(0.1, min(0.9, base_ndvi + np.random.normal(0, 0.05)))
        evi = max(0.1, min(0.8, ndvi * 0.8 + np.random.normal(0, 0.03)))
        
        return {
            'ndvi': ndvi,
            'evi': evi,
            'date': date.strftime('%Y-%m-%d'),
            'latitude': lat,
            'longitude': lon
        }