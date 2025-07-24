#!/usr/bin/env python
import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'personality_predictor.settings')
django.setup()

from ml_app.models import PersonalityData, TrainedModel

# Clear all data
PersonalityData.objects.all().delete()
TrainedModel.objects.all().delete()

print("Database cleared successfully!")
print(f"PersonalityData records: {PersonalityData.objects.count()}")
print(f"TrainedModel records: {TrainedModel.objects.count()}") 