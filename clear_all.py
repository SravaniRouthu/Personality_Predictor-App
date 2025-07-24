#!/usr/bin/env python
import os
import django
import shutil

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'personality_predictor.settings')
django.setup()

from ml_app.models import PersonalityData, TrainedModel
from django.db import connection

def clear_everything():
    """Clear all data and reset everything"""
    print("Clearing all data...")
    
    # Clear all data
    personality_count = PersonalityData.objects.count()
    model_count = TrainedModel.objects.count()
    
    PersonalityData.objects.all().delete()
    TrainedModel.objects.all().delete()
    
    # Reset auto-increment counters
    with connection.cursor() as cursor:
        cursor.execute("DELETE FROM sqlite_sequence WHERE name IN ('ml_app_personalitydata', 'ml_app_trainedmodel')")
    
    # Clear media files
    media_dir = 'media'
    if os.path.exists(media_dir):
        for item in os.listdir(media_dir):
            item_path = os.path.join(media_dir, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
        print("Media files cleared")
    
    print(f"Database cleared successfully!")
    print(f"Removed {personality_count} personality records")
    print(f"Removed {model_count} trained models")
    print(f"Current records: {PersonalityData.objects.count()}")
    print(f"Current models: {TrainedModel.objects.count()}")
    print("All data and files have been cleared!")

if __name__ == "__main__":
    clear_everything() 