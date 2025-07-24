from django.core.management.base import BaseCommand
from ml_app.models import PersonalityData, TrainedModel

class Command(BaseCommand):
    help = 'Clear all data from the database'

    def handle(self, *args, **options):
        # Clear all data
        personality_count = PersonalityData.objects.count()
        model_count = TrainedModel.objects.count()
        
        PersonalityData.objects.all().delete()
        TrainedModel.objects.all().delete()
        
        self.stdout.write(
            self.style.SUCCESS(
                f'Successfully cleared database! Removed {personality_count} personality records and {model_count} trained models.'
            )
        ) 