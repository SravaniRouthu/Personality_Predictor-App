#!/usr/bin/env python
"""
Check database status for Personality Predictor Django app
"""
import os
import sys
import django

# Add the project directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Set up Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'personality_predictor.settings')
django.setup()

from django.db import connection
from ml_app.models import PersonalityData, TrainedModel

def check_database():
    """Check database status"""
    print("🔍 Checking database status...")
    
    # Check if tables exist
    try:
        with connection.cursor() as cursor:
            cursor.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name LIKE 'ml_app_%'
            """)
            tables = cursor.fetchall()
            print(f"✅ Found {len(tables)} ml_app tables: {[table[0] for table in tables]}")
    except Exception as e:
        print(f"❌ Error checking tables: {e}")
        return False
    
    # Check data records
    try:
        data_count = PersonalityData.objects.count()
        print(f"📊 PersonalityData records: {data_count}")
        
        if data_count > 0:
            # Show first few records
            records = PersonalityData.objects.all()[:3]
            for i, record in enumerate(records):
                print(f"  Record {i+1}: ID={record.id}, Personality={record.personality}")
    except Exception as e:
        print(f"❌ Error checking data: {e}")
    
    # Check model records
    try:
        model_count = TrainedModel.objects.count()
        print(f"🤖 TrainedModel records: {model_count}")
        
        if model_count > 0:
            # Show first few models
            models = TrainedModel.objects.all()[:3]
            for i, model in enumerate(models):
                print(f"  Model {i+1}: {model.name} ({model.model_type}) - Accuracy: {model.accuracy}")
    except Exception as e:
        print(f"❌ Error checking models: {e}")
    
    print("🎉 Database check completed!")
    return True

if __name__ == '__main__':
    check_database() 