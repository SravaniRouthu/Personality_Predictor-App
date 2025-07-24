from django import forms
from .models import PersonalityData, TrainedModel

class PersonalityPredictionForm(forms.Form):
    """Form for individual personality prediction"""
    time_spent_alone = forms.IntegerField(
        min_value=0, max_value=24,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Hours spent alone per day (0-24)'
        }),
        help_text="How many hours do you spend alone per day?"
    )
    
    stage_fear = forms.ChoiceField(
        choices=[('Yes', 'Yes'), ('No', 'No')],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text="Do you experience stage fear?"
    )
    
    social_event_attendance = forms.IntegerField(
        min_value=0, max_value=20,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of social events attended per month'
        }),
        help_text="How many social events do you attend per month?"
    )
    
    going_outside = forms.IntegerField(
        min_value=0, max_value=20,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Times going outside per week'
        }),
        help_text="How many times do you go outside per week?"
    )
    
    drained_after_socializing = forms.ChoiceField(
        choices=[('Yes', 'Yes'), ('No', 'No')],
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text="Do you feel drained after socializing?"
    )
    
    friends_circle_size = forms.IntegerField(
        min_value=0, max_value=50,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Number of close friends'
        }),
        help_text="How many close friends do you have?"
    )
    
    post_frequency = forms.IntegerField(
        min_value=0, max_value=20,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'Social media posts per week'
        }),
        help_text="How often do you post on social media per week?"
    )
    
    model_choice = forms.ModelChoiceField(
        queryset=TrainedModel.objects.all(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text="Select the model to use for prediction"
    )

class CSVUploadForm(forms.Form):
    """Form for bulk CSV prediction"""
    csv_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        }),
        help_text="Upload a CSV file with personality data for bulk predictions"
    )
    
    model_choice = forms.ModelChoiceField(
        queryset=TrainedModel.objects.all(),
        widget=forms.Select(attrs={'class': 'form-control'}),
        help_text="Select the model to use for predictions"
    )

class ModelTrainingForm(forms.Form):
    """Form for model training with choice"""
    MODEL_CHOICES = [
        ('logistic_regression', '🤖 Logistic Regression Only'),
        ('random_forest', '🌳 Random Forest Only'),
        ('both', '🔄 Compare Both Models (Recommended)')
    ]
    
    model_choice = forms.ChoiceField(
        choices=MODEL_CHOICES,
        widget=forms.Select(attrs={'class': 'form-control'}),
        initial='both',
        help_text="Choose which model(s) to train. Comparing both gives you better insights."
    )
    
    model_name = forms.CharField(
        max_length=100,
        widget=forms.TextInput(attrs={
            'class': 'form-control',
            'placeholder': 'Enter a name for your models (e.g., MyPersonalityModel)'
        }),
        help_text="Give your models a descriptive name"
    )

class DataUploadForm(forms.Form):
    """Form for uploading dataset"""
    csv_file = forms.FileField(
        widget=forms.FileInput(attrs={
            'class': 'form-control',
            'accept': '.csv'
        }),
        help_text="Upload your personality dataset CSV file"
    )
    
    clear_existing = forms.BooleanField(
        required=False,
        initial=True,
        widget=forms.CheckboxInput(attrs={
            'class': 'form-check-input',
            'id': 'clear_existing'
        }),
        help_text="✅ RECOMMENDED: Clear existing data before importing (prevents duplicates)"
    ) 