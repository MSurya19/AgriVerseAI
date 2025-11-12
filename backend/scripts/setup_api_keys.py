#!/usr/bin/env python3
"""
API Key Setup Wizard for AgriVerseAI
Run this to set up all your API keys interactively
"""

import os
import sys
from dotenv import load_dotenv, set_key

def setup_api_keys():
    """Interactive API key setup wizard"""
    print("🔑 AgriVerseAI API Key Setup Wizard")
    print("=" * 50)
    
    # Load existing .env file
    env_path = '../../.env'
    load_dotenv(env_path)
    
    keys_setup = {}
    
    # USDA API Key
    print("\n🇺🇸 USDA API Key Setup")
    print("-" * 30)
    print("1. Visit: https://quickstats.nass.usda.gov/api/")
    print("2. Sign up for a free account")
    print("3. Get your API key from the dashboard")
    
    usda_key = input("Enter your USDA API key (or press Enter to skip): ").strip()
    if usda_key:
        set_key(env_path, 'USDA_API_KEY', usda_key)
        keys_setup['USDA'] = True
        print("✅ USDA API key saved!")
    else:
        print("⚠️  USDA API key skipped - using sample data")
        keys_setup['USDA'] = False
    
    # India API Key
    print("\n🇮🇳 India Data API Key Setup")
    print("-" * 30)
    print("1. Visit: https://data.gov.in/")
    print("2. Register for a free account")
    print("3. Go to Developer section and generate API key")
    
    india_key = input("Enter your India API key (or press Enter to skip): ").strip()
    if india_key:
        set_key(env_path, 'INDIA_API_KEY', india_key)
        keys_setup['INDIA'] = True
        print("✅ India API key saved!")
    else:
        print("⚠️  India API key skipped - using sample data")
        keys_setup['INDIA'] = False
    
    # NASA POWER (no key needed)
    print("\n🌍 NASA POWER API")
    print("-" * 30)
    print("✅ No API key needed! NASA POWER is completely free.")
    keys_setup['NASA'] = True
    
    # Summary
    print("\n📋 Setup Summary:")
    print("=" * 30)
    for service, status in keys_setup.items():
        status_icon = "✅" if status else "⚠️ "
        print(f"{status_icon} {service}: {'Real Data' if status else 'Sample Data'}")
    
    print(f"\n🎉 Setup complete! Your API keys are saved in: {env_path}")
    print("\n💡 Next steps:")
    print("1. Run: python download_usda_with_key.py")
    print("2. Run: python download_india_with_key.py") 
    print("3. Run: python train_yield_model.py")

if __name__ == "__main__":
    setup_api_keys()