import json
import numpy as np
from PIL import Image
import io
import base64
import requests
from datetime import datetime

class AgriHelpers:
    @staticmethod
    def load_config():
        """Load configuration from JSON file"""
        try:
            with open('config.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "model_paths": {
                    "yield_model": "../models/yield_model_final.h5",
                    "disease_model": "../models/disease_model_final.h5", 
                    "yield_scaler": "../models/yield_scaler.pkl",
                    "class_indices": "../models/class_indices.json"
                },
                "api_settings": {
                    "host": "0.0.0.0",
                    "port": 5000,
                    "debug": True
                },
                "data_sources": {
                    "nasa_power_url": "https://power.larc.nasa.gov/api/temporal/daily/point",
                    "earth_engine_credentials": "service_account.json"
                }
            }
    
    @staticmethod
    def image_to_base64(image):
        """Convert PIL Image to base64 string for API responses"""
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=85)
        return base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    @staticmethod
    def base64_to_image(base64_string):
        """Convert base64 string to PIL Image"""
        image_data = base64.b64decode(base64_string)
        return Image.open(io.BytesIO(image_data))
    
    @staticmethod
    def calculate_confidence(predictions):
        """Calculate confidence score from model predictions"""
        if len(predictions.shape) > 1:
            return float(np.max(predictions, axis=1)[0])
        return float(np.max(predictions))
    
    @staticmethod
    def get_treatment_recommendations(disease_name, severity="Medium"):
        """Get AI-powered treatment recommendations for plant diseases"""
        recommendations = {
            "Apple___Apple_scab": {
                "Low": [
                    "Apply preventive fungicides containing myclobutanil",
                    "Improve air circulation through proper pruning",
                    "Remove fallen leaves in autumn"
                ],
                "Medium": [
                    "Apply curative fungicides like captan or sulfur",
                    "Remove and destroy visibly infected leaves",
                    "Apply fungicides every 7-10 days during spring"
                ],
                "High": [
                    "Apply systemic fungicides immediately",
                    "Remove severely infected trees to prevent spread",
                    "Implement strict sanitation practices"
                ]
            },
            "Tomato___Early_blight": {
                "Low": [
                    "Apply copper-based fungicides preventively",
                    "Water at soil level to avoid wetting leaves",
                    "Ensure proper plant spacing"
                ],
                "Medium": [
                    "Apply chlorothalonil or mancozeb fungicides",
                    "Remove lower infected leaves regularly",
                    "Apply mulch to prevent soil splash"
                ],
                "High": [
                    "Apply systemic fungicides like azoxystrobin",
                    "Remove severely infected plants",
                    "Rotate crops next season"
                ]
            },
            "Corn___Common_rust": {
                "Low": [
                    "Plant resistant varieties next season",
                    "Ensure proper nitrogen fertilization",
                    "Monitor field regularly"
                ],
                "Medium": [
                    "Apply fungicides containing pyraclostrobin",
                    "Remove alternative host plants",
                    "Improve field drainage"
                ],
                "High": [
                    "Apply combination fungicides immediately",
                    "Consider early harvest if severe",
                    "Destroy crop residue after harvest"
                ]
            },
            "Healthy": {
                "Low": [
                    "Continue regular monitoring",
                    "Maintain proper irrigation schedule",
                    "Apply balanced fertilization"
                ],
                "Medium": [
                    "Implement preventive fungicide application",
                    "Monitor weather conditions for disease risk",
                    "Maintain plant health records"
                ],
                "High": [
                    "Increase monitoring frequency",
                    "Prepare contingency plan for disease outbreak",
                    "Keep emergency contact for agricultural expert"
                ]
            }
        }
        
        # Default recommendations for unknown diseases
        default_recs = {
            "Low": [
                "Consult local agricultural extension officer",
                "Monitor plant health regularly",
                "Maintain proper growing conditions"
            ],
            "Medium": [
                "Isolate affected plants immediately",
                "Apply broad-spectrum fungicide",
                "Improve overall plant nutrition"
            ],
            "High": [
                "Seek immediate expert consultation",
                "Remove and destroy severely infected plants",
                "Implement strict quarantine measures"
            ]
        }
        
        disease_recs = recommendations.get(disease_name, default_recs)
        return disease_recs.get(severity, disease_recs["Medium"])
    
    @staticmethod
    def calculate_yield_impact(disease_present, disease_severity, crop_type):
        """Calculate potential yield impact based on disease presence and severity"""
        base_impact = {
            "wheat": {"Low": 0.05, "Medium": 0.15, "High": 0.35},
            "rice": {"Low": 0.08, "Medium": 0.18, "High": 0.40},
            "corn": {"Low": 0.06, "Medium": 0.16, "High": 0.38},
            "soybean": {"Low": 0.07, "Medium": 0.17, "High": 0.42}
        }
        
        if not disease_present:
            return 0.0
        
        crop_impact = base_impact.get(crop_type, base_impact["wheat"])
        return crop_impact.get(disease_severity, crop_impact["Medium"])
    
    @staticmethod
    def generate_farm_insights(yield_prediction, disease_analysis, weather_data):
        """Generate comprehensive farm insights"""
        insights = []
        
        # Yield-based insights
        if yield_prediction > 5000:
            insights.append("📈 Excellent yield potential detected")
        elif yield_prediction > 3500:
            insights.append("✅ Good yield potential with proper management")
        else:
            insights.append("⚠️ Below average yield potential - consider interventions")
        
        # Disease-based insights
        if disease_analysis.get('has_disease', False):
            severity = disease_analysis.get('severity', 'Medium')
            if severity == 'High':
                insights.append("🚨 High disease severity - immediate action required")
            elif severity == 'Medium':
                insights.append("⚠️ Moderate disease presence - monitor closely")
            else:
                insights.append("🔍 Low disease risk - continue monitoring")
        else:
            insights.append("🌱 No significant diseases detected")
        
        # Weather-based insights
        if weather_data.get('rainfall', 0) > 1000:
            insights.append("💧 High rainfall expected - watch for fungal diseases")
        elif weather_data.get('rainfall', 0) < 300:
            insights.append("🌵 Low rainfall predicted - ensure irrigation")
        
        if weather_data.get('temperature', 25) > 35:
            insights.append("🔥 High temperatures expected - monitor water stress")
        
        return insights
    
    @staticmethod
    def validate_yield_input(data):
        """Validate yield prediction input data"""
        errors = []
        
        required_fields = ['ndvi', 'evi', 'rainfall', 'temperature', 'crop_type']
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        if 'ndvi' in data and not (0 <= data['ndvi'] <= 1):
            errors.append("NDVI must be between 0 and 1")
        
        if 'evi' in data and not (0 <= data['evi'] <= 1):
            errors.append("EVI must be between 0 and 1")
        
        if 'rainfall' in data and data['rainfall'] < 0:
            errors.append("Rainfall cannot be negative")
        
        if 'temperature' in data and not (-50 <= data['temperature'] <= 60):
            errors.append("Temperature must be between -50°C and 60°C")
        
        valid_crops = ['wheat', 'rice', 'corn', 'soybean']
        if 'crop_type' in data and data['crop_type'] not in valid_crops:
            errors.append(f"Crop type must be one of: {', '.join(valid_crops)}")
        
        return errors
    
    @staticmethod
    def format_timestamp(timestamp=None):
        """Format timestamp for consistent API responses"""
        if timestamp is None:
            timestamp = datetime.now()
        return timestamp.isoformat()

# Create singleton instance
agri_helpers = AgriHelpers()