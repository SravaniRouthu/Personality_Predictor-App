#!/usr/bin/env python
"""
Force create database tables for Personality Predictor Django app
"""
import os
import sys
import django

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'personality_predictor.settings')
django.setup()

from django.core.management import execute_from_command_line
from django.db import connection
from django.db import models
from ml_app.models import PersonalityData, TrainedModel, PredictionResult

def create_tables():
    """Force create the database tables"""
    print("Creating database tables...")
    
    try:
        # Create tables using Django's table creation
        with connection.schema_editor() as schema_editor:
            # Create PersonalityData table
            schema_editor.create_model(PersonalityData)
            print("✅ Created PersonalityData table")
            
            # Create TrainedModel table
            schema_editor.create_model(TrainedModel)
            print("✅ Created TrainedModel table")
            
            # Create PredictionResult table
            schema_editor.create_model(PredictionResult)
            print("✅ Created PredictionResult table")
        
        print("🎉 All tables created successfully!")
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

if __name__ == '__main__':
    create_tables() 