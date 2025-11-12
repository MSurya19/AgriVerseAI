import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import os

class EnhancedYieldDataCollector:
    def __init__(self):
        self.satellite_data_cache = {}
        self.climate_data_cache = {}
        
    def generate_realistic_yield_data(self, num_samples=10000):
        """Generate realistic synthetic yield data based on agricultural research"""
        print("🌱 Generating realistic yield training data...")
        
        np.random.seed(42)
        data = []
        
        # Base yields for different crops (kg/hectare)
        base_yields = {
            'wheat': {'min': 2500, 'max': 6000, 'optimal_temp': 20, 'optimal_rainfall': 500},
            'rice': {'min': 3000, 'max': 7000, 'optimal_temp': 25, 'optimal_rainfall': 1000},
            'corn': {'min': 4000, 'max': 10000, 'optimal_temp': 22, 'optimal_rainfall': 600},
            'soybean': {'min': 2000, 'max': 4000, 'optimal_temp': 24, 'optimal_rainfall': 500}
        }
        
        for i in range(num_samples):
            crop_type = np.random.choice(list(base_yields.keys()))
            crop_info = base_yields[crop_type]
            
            # Generate realistic agricultural parameters
            ndvi = np.random.normal(0.7, 0.15)  # Normal distribution around 0.7
            ndvi = max(0.1, min(0.9, ndvi))
            
            evi = np.random.normal(0.5, 0.1)  # EVI typically lower than NDVI
            evi = max(0.1, min(0.8, evi))
            
            # Rainfall based on crop requirements with seasonal variation
            base_rainfall = crop_info['optimal_rainfall']
            rainfall = max(100, np.random.normal(base_rainfall, base_rainfall * 0.3))
            
            # Temperature with seasonal variation
            base_temp = crop_info['optimal_temp']
            temperature = np.random.normal(base_temp, 5)
            temperature = max(10, min(40, temperature))
            
            humidity = np.random.normal(65, 15)
            humidity = max(30, min(95, humidity))
            
            soil_ph = np.random.normal(6.5, 0.5)  # Most crops prefer slightly acidic to neutral
            soil_ph = max(4.5, min(8.5, soil_ph))
            
            nitrogen_level = np.random.normal(60, 20)  # kg/hectare
            nitrogen_level = max(10, min(150, nitrogen_level))
            
            # Calculate realistic yield based on scientific models
            base_yield = np.random.uniform(crop_info['min'], crop_info['max'])
            
            # Yield factors (based on agricultural research)
            ndvi_factor = 0.3 * (ndvi / 0.7)  # NDVI contribution
            rainfall_factor = 0.2 * (1 - abs(rainfall - crop_info['optimal_rainfall']) / crop_info['optimal_rainfall'])
            temp_factor = 0.15 * (1 - abs(temperature - crop_info['optimal_temp']) / 15)
            humidity_factor = 0.1 * (humidity / 70)
            soil_ph_factor = 0.1 * (1 - abs(soil_ph - 6.5) / 2)
            nitrogen_factor = 0.15 * (nitrogen_level / 60)
            
            # Combine factors with some randomness
            total_factor = (ndvi_factor + rainfall_factor + temp_factor + 
                          humidity_factor + soil_ph_factor + nitrogen_factor)
            
            # Apply factors with some random variation
            yield_value = base_yield * (0.7 + 0.3 * total_factor) * np.random.normal(1, 0.1)
            yield_value = max(1000, yield_value)  # Minimum yield
            
            data.append({
                'ndvi': ndvi,
                'evi': evi,
                'rainfall': rainfall,
                'temperature': temperature,
                'humidity': humidity,
                'crop_type': crop_type,
                'soil_ph': soil_ph,
                'nitrogen_level': nitrogen_level,
                'yield': yield_value
            })
        
        return pd.DataFrame(data)
    
    def get_satellite_data_simulated(self, lat, lon, start_date, end_date):
        """Simulate Sentinel-2 satellite data for NDVI/EVI"""
        print(f"📡 Generating satellite data for ({lat}, {lon})")
        
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        data = []
        
        for date in dates:
            # Seasonal NDVI pattern (growing season variation)
            day_of_year = date.timetuple().tm_yday
            
            if 60 <= day_of_year <= 240:  # Growing season
                base_ndvi = 0.6 + 0.2 * np.sin(2 * np.pi * (day_of_year - 100) / 180)
            else:  # Off-season
                base_ndvi = 0.3 + 0.1 * np.sin(2 * np.pi * (day_of_year - 100) / 180)
            
            # Add random variation and noise
            ndvi = max(0.1, min(0.9, base_ndvi + np.random.normal(0, 0.05)))
            evi = max(0.1, min(0.8, ndvi * 0.8 + np.random.normal(0, 0.03)))
            
            data.append({
                'date': date.strftime('%Y-%m-%d'),
                'latitude': lat,
                'longitude': lon,
                'ndvi': ndvi,
                'evi': evi,
                'quality_score': np.random.uniform(0.8, 1.0)
            })
        
        return data
    
    def get_soil_data_simulated(self, lat, lon):
        """Simulate soil data based on location"""
        # Different soil types in different regions
        soil_types = {
            'clay': {'ph_range': (5.5, 7.0), 'nitrogen_range': (40, 80)},
            'loam': {'ph_range': (6.0, 7.5), 'nitrogen_range': (50, 100)},
            'sandy': {'ph_range': (5.0, 6.5), 'nitrogen_range': (20, 60)}
        }
        
        # Simple geographic pattern
        if lat > 35:  # Northern regions
            soil_type = 'clay'
        elif lat < 20:  # Southern regions
            soil_type = 'sandy'
        else:  # Middle regions
            soil_type = 'loam'
        
        soil_info = soil_types[soil_type]
        
        return {
            'latitude': lat,
            'longitude': lon,
            'soil_type': soil_type,
            'soil_ph': np.random.uniform(*soil_info['ph_range']),
            'nitrogen_level': np.random.uniform(*soil_info['nitrogen_range']),
            'organic_matter': np.random.uniform(1.5, 4.0),
            'phosphorus': np.random.uniform(20, 80),
            'potassium': np.random.uniform(150, 300)
        }
    
    def save_training_dataset(self, file_path='../../datasets/yield_data/training_data.csv'):
        """Generate and save comprehensive training dataset"""
        df = self.generate_realistic_yield_data(10000)
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        # Save to CSV
        df.to_csv(file_path, index=False)
        print(f"✅ Saved training data to {file_path}")
        print(f"📊 Dataset contains {len(df)} samples")
        print(f"🌾 Crop distribution:")
        print(df['crop_type'].value_counts())
        
        return df

if __name__ == "__main__":
    collector = EnhancedYieldDataCollector()
    
    # Generate and save training data
    df = collector.save_training_dataset()
    
    # Print dataset statistics
    print("\n📈 Dataset Statistics:")
    print(f"Average Yield: {df['yield'].mean():.2f} kg/ha")
    print(f"Yield Range: {df['yield'].min():.2f} - {df['yield'].max():.2f} kg/ha")
    print(f"Average NDVI: {df['ndvi'].mean():.3f}")
    print(f"Average Rainfall: {df['rainfall'].mean():.1f} mm")