#!/usr/bin/env python
"""
Database setup script for Personality Predictor Django app
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

def setup_database():
    """Set up the database with all required tables"""
    print("Setting up database...")
    
    # Run migrations
    try:
        execute_from_command_line(['manage.py', 'migrate'])
        print("✅ Migrations completed successfully!")
    except Exception as e:
        print(f"❌ Error running migrations: {e}")
        return False
    
    # Verify tables exist
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
    
    print("🎉 Database setup completed successfully!")
    return True

if __name__ == '__main__':
    setup_database() 