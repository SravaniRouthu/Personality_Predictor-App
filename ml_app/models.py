from django.db import models
import json

class PersonalityData(models.Model):
    # Fixed fields for backward compatibility
    time_spent_alone = models.FloatField(default=0)
    stage_fear = models.CharField(max_length=10, default='No')
    social_event_attendance = models.FloatField(default=0)
    going_outside = models.FloatField(default=0)
    drained_after_socializing = models.CharField(max_length=10, default='No')
    friends_circle_size = models.FloatField(default=0)
    post_frequency = models.FloatField(default=0)
    personality = models.CharField(max_length=20)
    
    # Flexible storage for any dataset
    feature_data = models.JSONField(default=dict, blank=True)
    target_value = models.CharField(max_length=50, blank=True, default='')
    
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Personality Data"
        verbose_name_plural = "Personality Data"
    
    def __str__(self):
        return f"Record {self.id} - {self.personality}"
    
    def get_features_dict(self):
        """Get features as dictionary"""
        if self.feature_data:
            return self.feature_data
        else:
            # Fallback to fixed fields
            return {
                'time_spent_alone': self.time_spent_alone,
                'stage_fear': self.stage_fear,
                'social_event_attendance': self.social_event_attendance,
                'going_outside': self.going_outside,
                'drained_after_socializing': self.drained_after_socializing,
                'friends_circle_size': self.friends_circle_size,
                'post_frequency': self.post_frequency
            }
    
    def get_target(self):
        """Get target value"""
        return self.target_value if self.target_value else self.personality

class TrainedModel(models.Model):
    MODEL_TYPES = [
        ('logistic_regression', 'Logistic Regression'),
        ('random_forest', 'Random Forest'),
        ('svm', 'Support Vector Machine'),
        ('neural_network', 'Neural Network'),
    ]
    
    name = models.CharField(max_length=100)
    model_type = models.CharField(max_length=20, choices=MODEL_TYPES)
    model_file = models.FileField(upload_to='models/')
    accuracy = models.FloatField()
    precision = models.FloatField()
    recall = models.FloatField()
    f1_score = models.FloatField()
    training_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Trained Model"
        verbose_name_plural = "Trained Models"
    
    def __str__(self):
        return f"{self.name} ({self.model_type})"

class PredictionResult(models.Model):
    model_used = models.ForeignKey(TrainedModel, on_delete=models.CASCADE)
    input_features = models.JSONField(default=dict)
    prediction = models.CharField(max_length=20)
    confidence = models.FloatField()
    probabilities = models.JSONField(default=dict)
    prediction_date = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Prediction Result"
        verbose_name_plural = "Prediction Results"
    
    def __str__(self):
        return f"Prediction {self.id} - {self.prediction}" 