import os
import django

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'personality_predictor.settings')
django.setup()

from ml_app.models import PersonalityData, TrainedModel

# Clear all data
personality_count = PersonalityData.objects.count()
model_count = TrainedModel.objects.count()

PersonalityData.objects.all().delete()
TrainedModel.objects.all().delete()

print(f"Database cleared successfully!")
print(f"Removed {personality_count} personality records")
print(f"Removed {model_count} trained models")
print(f"Current records: {PersonalityData.objects.count()}")
print(f"Current models: {TrainedModel.objects.count()}") 