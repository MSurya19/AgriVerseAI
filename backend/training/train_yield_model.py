import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, CSVLogger
import numpy as np
import json
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import sys
import matplotlib.pyplot as plt

sys.path.append('..')
from models.yield_model import create_yield_model
from data.yield_data_collector import EnhancedYieldDataCollector

def plot_training_history(history, save_path='../../logs/yield_training_history.png'):
    """Plot training history for model evaluation"""
    plt.figure(figsize=(12, 4))
    
    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot MAE
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (kg/ha)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Training plots saved to {save_path}")

def evaluate_model_performance(model, X_test, y_test, scaler):
    """Comprehensive model evaluation"""
    # Make predictions
    y_pred = model.predict(X_test).flatten()
    
    # Calculate metrics
    mae = np.mean(np.abs(y_pred - y_test))
    mse = np.mean((y_pred - y_test) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    # R-squared
    ss_res = np.sum((y_test - y_pred) ** 2)
    ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"📊 Model Performance Metrics:")
    print(f"   MAE: {mae:.2f} kg/ha")
    print(f"   RMSE: {rmse:.2f} kg/ha") 
    print(f"   MAPE: {mape:.2f}%")
    print(f"   R² Score: {r2:.4f}")
    
    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2': r2
    }

def train_yield_model():
    """Enhanced yield model training with real data support"""
    print("🚀 Starting enhanced yield model training...")
    
    # Create directories
    os.makedirs('../../models', exist_ok=True)
    os.makedirs('../../logs', exist_ok=True)
    os.makedirs('../../datasets/yield_data', exist_ok=True)
    
    # Check for existing real data or generate synthetic data
    data_file = '../../datasets/yield_data/training_data.csv'
    
    if os.path.exists(data_file):
        print("📁 Loading existing yield data...")
        df = pd.read_csv(data_file)
    else:
        print("🔄 Generating realistic synthetic yield data...")
        collector = EnhancedYieldDataCollector()
        df = collector.save_training_dataset(data_file)
    
    print(f"📊 Dataset loaded: {len(df)} samples")
    print(f"🌾 Crop distribution:")
    print(df['crop_type'].value_counts())
    
    # Convert crop type to numeric
    crop_mapping = {'wheat': 0, 'rice': 1, 'corn': 2, 'soybean': 3}
    df['crop_type_numeric'] = df['crop_type'].map(crop_mapping)
    
    # Prepare features and target
    feature_columns = ['ndvi', 'evi', 'rainfall', 'temperature', 'humidity', 
                      'crop_type_numeric', 'soil_ph', 'nitrogen_level']
    
    X = df[feature_columns].values
    y = df['yield'].values
    
    print(f"🎯 Features: {feature_columns}")
    print(f"🎯 Target: Yield (kg/ha)")
    print(f"📈 Data shape: X={X.shape}, y={y.shape}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Reshape for LSTM (samples, time steps, features)
    X_reshaped = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42, stratify=df['crop_type_numeric']
    )
    
    print(f"📚 Training samples: {X_train.shape[0]}")
    print(f"🧪 Test samples: {X_test.shape[0]}")
    
    # Create and compile model
    model = create_yield_model(sequence_length=1, feature_count=len(feature_columns))
    
    # Callbacks
    callbacks = [
        EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
        ModelCheckpoint('../../models/yield_model_best.h5', save_best_only=True, verbose=1),
        ReduceLROnPlateau(factor=0.5, patience=10, verbose=1),
        CSVLogger('../../logs/yield_training_log.csv')
    ]
    
    # Train model
    print("📈 Training yield prediction model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Save final model and scaler
    model.save('../../models/yield_model_final.h5')
    joblib.dump(scaler, '../../models/yield_scaler.pkl')
    
    # Save crop mapping
    with open('../../models/crop_mapping.json', 'w') as f:
        json.dump(crop_mapping, f, indent=4)
    
    # Evaluate model
    print("\n📊 Model Evaluation:")
    metrics = evaluate_model_performance(model, X_test, y_test, scaler)
    
    # Plot training history
    plot_training_history(history)
    
    # Save metrics
    with open('../../logs/yield_model_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print(f"✅ Training completed!")
    print(f"📁 Models saved in: ../models/")
    print(f"📊 Logs saved in: ../logs/")
    
    return history, metrics

if __name__ == "__main__":
    train_yield_model()