from django.shortcuts import render, redirect, get_object_or_404
from django.contrib import messages
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.core.files.storage import default_storage
from django.conf import settings
from django.db.models import Count
from django.db import connection
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
from .models import PersonalityData, TrainedModel, PredictionResult
from .forms import PersonalityPredictionForm, DataUploadForm
from .ml_engine import PersonalityMLEngine
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.utils
from sklearn.metrics import accuracy_score

def home(request):
    """Home page with status overview"""
    total_records, trained_models = get_database_status()
    
    # Get additional stats
    accuracy = "N/A"
    predictions_made = 0
    
    try:
        # Get accuracy from the best model
        best_model = TrainedModel.objects.order_by('-accuracy').first()
        if best_model:
            accuracy = f"{best_model.accuracy:.1f}%"
        
        # Get prediction count
        predictions_made = PredictionResult.objects.count()
    except Exception as e:
        print(f"Error getting additional stats: {e}")
    
    context = {
        'total_records': total_records,
        'trained_models': trained_models,
        'accuracy': accuracy,
        'predictions_made': predictions_made,
    }
    return render(request, 'ml_app/home.html', context)

def clear_database():
    """Clear all database records and reset counters"""
    try:
        # Delete all records
        PersonalityData.objects.all().delete()
        TrainedModel.objects.all().delete()
        PredictionResult.objects.all().delete()
        
        # Reset SQLite auto-increment counters
        from django.db import connection
        with connection.cursor() as cursor:
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='ml_app_personalitydata'")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='ml_app_trainedmodel'")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='ml_app_predictionresult'")
        
        # Clear model files
        model_dir = settings.MEDIA_ROOT / 'models'
        if model_dir.exists():
            for file in model_dir.glob('*'):
                if file.is_file():
                    file.unlink()
        
        print("Database cleared successfully!")
        return True
    except Exception as e:
        print(f"Error clearing database: {e}")
        return False

def reset_application(request):
    """Reset the entire application - clear database and model files"""
    if request.method == 'POST':
        success = clear_database()
        if success:
            messages.success(request, "✅ Application reset successfully! All data and models have been cleared.")
        else:
            messages.error(request, "❌ Error resetting application. Please try again.")
        return redirect('home')
    
    return render(request, 'ml_app/reset_application.html')

def get_database_status():
    """Get current database status with error handling"""
    try:
        # Use raw SQL to get accurate counts
        from django.db import connection
        with connection.cursor() as cursor:
            # Check PersonalityData count
            cursor.execute("SELECT COUNT(*) FROM ml_app_personalitydata")
            data_count = cursor.fetchone()[0]
            
            # Check TrainedModel count
            cursor.execute("SELECT COUNT(*) FROM ml_app_trainedmodel")
            model_count = cursor.fetchone()[0]
        
        print(f"Database status: {data_count} records, {model_count} models")
        return data_count, model_count
    except Exception as e:
        print(f"Database error in get_database_status: {e}")
        # Fallback to Django ORM
        try:
            data_count = PersonalityData.objects.count()
            model_count = TrainedModel.objects.count()
            print(f"Fallback counts: {data_count} records, {model_count} models")
            return data_count, model_count
        except Exception as e2:
            print(f"Fallback also failed: {e2}")
            return 0, 0

def upload_data(request):
    """Upload data page"""
    if request.method == 'POST':
        try:
            # Clear existing data and models for fresh start
            clear_database()
            
            # Get uploaded file
            csv_file = request.FILES.get('csv_file')
            if not csv_file:
                messages.error(request, "Please select a CSV file to upload.")
                return render(request, 'ml_app/upload_data.html')
            
            # Save file temporarily
            file_path = default_storage.save(f'temp/{csv_file.name}', csv_file)
            full_path = default_storage.path(file_path)
            
            # Read CSV file
            df = pd.read_csv(full_path)
            
            # Validate that we have at least 2 columns (features + target)
            if len(df.columns) < 2:
                messages.error(request, "CSV file must have at least 2 columns (features + target).")
                return render(request, 'ml_app/upload_data.html')
            
            # Get column names from the actual data
            column_names = list(df.columns)
            target_column = column_names[-1]  # Last column is target
            feature_columns = column_names[:-1]  # All other columns are features
            
            # Analyze the dataset
            total_rows = len(df)
            feature_count = len(feature_columns)
            target_values = df[target_column].unique()
            
            # Show dataset analysis
            analysis_message = f"""
            Dataset Analysis:
            - Total records: {total_rows}
            - Features: {feature_count} columns ({', '.join(feature_columns)})
            - Target: {target_column} with values: {', '.join(map(str, target_values))}
            - This is a binary classification problem
            """
            messages.info(request, analysis_message)
            
            # Store data with flexible approach
            records_to_create = []
            for _, row in df.iterrows():
                # Store all feature data as JSON for flexibility
                feature_data = {}
                for col in feature_columns:
                    feature_data[col] = row[col]
                
                record = PersonalityData(
                    personality=row[target_column],  # Use the actual target column
                    feature_data=feature_data,
                    target_value=row[target_column]
                )
                records_to_create.append(record)
            
            # Bulk create all records
            PersonalityData.objects.bulk_create(records_to_create)
            
            # Clean up temporary file
            default_storage.delete(file_path)
            
            # Get updated status
            data_count, _ = get_database_status()
            
            messages.success(request, f"✅ Dataset uploaded successfully! {data_count} records ready for training.")
            return redirect('train_models')
            
        except Exception as e:
            messages.error(request, f"Error uploading data: {str(e)}")
            return render(request, 'ml_app/upload_data.html')
    
    return render(request, 'ml_app/upload_data.html')

def train_models(request):
    """Train ML models page"""
    total_records = PersonalityData.objects.count()
    training_results = {}
    
    if request.method == 'POST':
        try:
            # Check if we have data to train on
            if total_records == 0:
                messages.error(request, "No data available for training. Please upload a dataset first.")
                return render(request, 'ml_app/train_models.html', {'total_records': total_records})
            
            # Get selected models and model names
            train_lr = request.POST.get('train_lr') == 'on'
            train_rf = request.POST.get('train_rf') == 'on'
            
            # Get model names
            lr_model_name = request.POST.get('lr_model_name', 'Logistic Regression')
            rf_model_name = request.POST.get('rf_model_name', 'Random Forest')
            
            if not train_lr and not train_rf:
                messages.error(request, "Please select at least one model to train.")
                return render(request, 'ml_app/train_models.html', {'total_records': total_records})
            
            # Initialize ML engine
            ml_engine = PersonalityMLEngine()
            
            # Load data from database and convert to DataFrame
            data_records = PersonalityData.objects.all()
            data_list = []
            for record in data_records:
                # Use the flexible feature data
                row_data = record.feature_data.copy()
                # Add target column with a proper name
                row_data['Personality'] = record.target_value  # Use consistent target column name
                data_list.append(row_data)
            
            df = pd.DataFrame(data_list)
            
            # Debug: Print data info
            print(f"DataFrame shape: {df.shape}")
            print(f"DataFrame columns: {df.columns.tolist()}")
            print(f"DataFrame head:\n{df.head()}")
            print(f"Target value counts:\n{df.iloc[:, -1].value_counts()}")
            
            # Preprocess data
            df_processed = ml_engine.preprocess_data_from_df(df)
            X, y = ml_engine.prepare_features(df_processed)
            
            # Debug: Print processed data info
            print(f"Processed X shape: {X.shape}")
            print(f"Processed y shape: {y.shape}")
            print(f"Processed y value counts: {np.bincount(y)}")
            
            # Train selected models
            trained_models = []
            
            if train_lr:
                # Train Logistic Regression with detailed results
                lr_result = ml_engine.train_logistic_regression(X, y, "Logistic Regression")
                
                # Add training metrics to results
                lr_result['training_accuracy'] = accuracy_score(lr_result['y_train'], lr_result['model'].predict(lr_result['X_train'])) * 100
                lr_result['validation_accuracy'] = lr_result['accuracy'] * 100
                lr_result['learning_rate'] = 0.001  # Default learning rate
                lr_result['epochs'] = 1000  # Default epochs
                lr_result['convergence'] = lr_result['model'].n_iter_[0] if hasattr(lr_result['model'], 'n_iter_') else 1000
                lr_result['precision'] = lr_result['precision'] * 100
                lr_result['recall'] = lr_result['recall'] * 100
                lr_result['f1_score'] = lr_result['f1_score'] * 100
                
                # Save model file
                model_file_path = ml_engine.save_model(lr_result, 'logistic_regression', 'Intro vs Extro_LR')
                
                # Create database record
                TrainedModel.objects.create(
                    name=lr_model_name,
                    model_type='logistic_regression',
                    model_file=model_file_path,
                    accuracy=lr_result['accuracy'],
                    precision=lr_result['precision'],
                    recall=lr_result['recall'],
                    f1_score=lr_result['f1_score']
                )
                trained_models.append(lr_model_name)
                training_results[lr_model_name] = lr_result
            
            if train_rf:
                # Train Random Forest with detailed results
                rf_result = ml_engine.train_random_forest(X, y, "Random Forest")
                
                # Add training metrics to results
                rf_result['training_accuracy'] = accuracy_score(rf_result['y_train'], rf_result['model'].predict(rf_result['X_train'])) * 100
                rf_result['validation_accuracy'] = rf_result['accuracy'] * 100
                rf_result['learning_rate'] = 0.01  # Default learning rate for RF
                rf_result['epochs'] = 100  # Default epochs for RF
                rf_result['convergence'] = rf_result['model'].n_estimators
                rf_result['precision'] = rf_result['precision'] * 100
                rf_result['recall'] = rf_result['recall'] * 100
                rf_result['f1_score'] = rf_result['f1_score'] * 100
                
                # Save model file
                model_file_path = ml_engine.save_model(rf_result, 'random_forest', 'Intro vs Extro_RF')
                
                # Create database record
                TrainedModel.objects.create(
                    name=rf_model_name,
                    model_type='random_forest',
                    model_file=model_file_path,
                    accuracy=rf_result['accuracy'],
                    precision=rf_result['precision'],
                    recall=rf_result['recall'],
                    f1_score=rf_result['f1_score']
                )
                trained_models.append(rf_model_name)
                training_results[rf_model_name] = rf_result
            
            messages.success(request, f"Successfully trained {len(trained_models)} models: {', '.join(trained_models)}")
            
            # Return the page with training results
            return render(request, 'ml_app/train_models.html', {
                'total_records': total_records,
                'training_results': training_results
            })
            
        except Exception as e:
            messages.error(request, f"Error training models: {str(e)}")
            return render(request, 'ml_app/train_models.html', {'total_records': total_records})
    
    return render(request, 'ml_app/train_models.html', {'total_records': total_records})

def predict_personality(request):
    """Make personality predictions"""
    models = TrainedModel.objects.all()
    
    if request.method == 'POST':
        try:
            if not models.exists():
                messages.error(request, "No trained models available. Please train models first.")
                return render(request, 'ml_app/predict_personality.html', {'models': models})
            
            # Get selected model
            model_id = request.POST.get('model_id')
            if not model_id:
                messages.error(request, "Please select a model.")
                return render(request, 'ml_app/predict_personality.html', {'models': models})
            
            trained_model = get_object_or_404(TrainedModel, id=model_id)
            
            # Get input data and map to expected column names
            input_data = {}
            
            # Map form field names to expected column names
            field_mapping = {
                'feature_time_spent_alone': 'Time_spent_Alone',
                'feature_stage_fear': 'Stage_fear',
                'feature_social_event_attendance': 'Social_event_attendance',
                'feature_going_outside': 'Going_outside',
                'feature_drained_after_socializing': 'Drained_after_socializing',
                'feature_friends_circle_size': 'Friends_circle_size',
                'feature_post_frequency': 'Post_frequency'
            }
            
            print("=== PREDICTION DEBUG ===")
            print(f"POST data: {request.POST}")
            
            for form_field, expected_column in field_mapping.items():
                value = request.POST.get(form_field)
                print(f"Form field: {form_field}, Value: {value}")
                if value:
                    # Convert numeric fields
                    if expected_column in ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']:
                        try:
                            input_data[expected_column] = float(value)
                        except ValueError:
                            input_data[expected_column] = 0.0
                    else:
                        input_data[expected_column] = value
            
            print(f"Processed input_data: {input_data}")
            
            if not input_data:
                messages.error(request, "Please provide input features.")
                return render(request, 'ml_app/predict_personality.html', {'models': models})
            
            # Load model and make prediction
            model_data = PersonalityMLEngine().load_model(trained_model.model_file)
            prediction_result = PersonalityMLEngine().predict_single(model_data, input_data)
            
            print(f"Prediction result type: {type(prediction_result)}")
            print(f"Prediction result: {prediction_result}")
            
            # Unpack the prediction result
            if isinstance(prediction_result, tuple) and len(prediction_result) == 3:
                prediction, confidence, probabilities = prediction_result
            else:
                # Handle case where predict_single returns a different format
                prediction = prediction_result
                confidence = 0.5
                probabilities = {'Introvert': 0.5, 'Extrovert': 0.5}
            
            print(f"Final prediction: {prediction}")
            print(f"Confidence: {confidence}")
            print(f"Probabilities: {probabilities}")
            
            # Generate explanation
            prediction_str = str(prediction).lower()
            if prediction_str in ['introvert', 'intro', '0']:
                explanation = "You show introverted tendencies. This means you likely prefer solitary activities, need time to recharge after social interactions, and may feel more comfortable in smaller, familiar groups."
                qualities = "Great qualities of introverts include deep thinking, strong focus, creativity, and the ability to form meaningful relationships."
            else:
                explanation = "You show extroverted tendencies. This means you likely enjoy social interactions, gain energy from being around others, and may prefer group activities and external stimulation."
                qualities = "Great qualities of extroverts include strong communication skills, natural leadership, enthusiasm, and the ability to energize others."
            
            context = {
                'models': models,
                'selected_model': trained_model,
                'prediction': prediction,
                'confidence': confidence,
                'probabilities': probabilities,
                'explanation': explanation,
                'qualities': qualities,
                'input_data': input_data
            }
            
            print(f"Context being passed to template: {context}")
            return render(request, 'ml_app/prediction_result.html', context)
            
        except Exception as e:
            print(f"Exception in prediction: {str(e)}")
            import traceback
            traceback.print_exc()
            messages.error(request, f"Error making prediction: {str(e)}")
            return render(request, 'ml_app/predict_personality.html', {'models': models})
    
    return render(request, 'ml_app/predict_personality.html', {'models': models})

def data_exploration(request):
    """Data exploration and visualization"""
    try:
        data_count = PersonalityData.objects.count()
        if data_count == 0:
            messages.warning(request, "No data available for exploration. Please upload a dataset first.")
            return redirect('home')
        
        # Get data for visualization
        records = PersonalityData.objects.all()
        
        # Create personality distribution chart
        personality_counts = {}
        for record in records:
            personality = record.get_target()
            personality_counts[personality] = personality_counts.get(personality, 0) + 1
        
        # Create Plotly pie chart
        fig = go.Figure(data=[go.Pie(
            labels=list(personality_counts.keys()),
            values=list(personality_counts.values()),
            hole=0.3
        )])
        fig.update_layout(
            title="Personality Distribution",
            showlegend=True,
            height=400
        )
        
        # Convert to JSON for template
        personality_chart = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        
        context = {
            'data_count': data_count,
            'personality_chart': personality_chart,
            'personality_counts': personality_counts
        }
        
        return render(request, 'ml_app/data_exploration.html', context)
        
    except Exception as e:
        messages.error(request, f"Error exploring data: {str(e)}")
        return redirect('home')

def about(request):
    """About page"""
    return render(request, 'ml_app/about.html') 

def bulk_test(request):
    """Bulk testing with CSV file upload"""
    # Get available models
    trained_models = TrainedModel.objects.all()
    
    if request.method == 'POST':
        try:
            # Get uploaded file
            csv_file = request.FILES.get('test_csv')
            if not csv_file:
                messages.error(request, "Please select a CSV file to upload.")
                return render(request, 'ml_app/bulk_test.html', {'trained_models': trained_models})
            
            # Get selected model
            model_id = request.POST.get('model_id')
            if not model_id:
                messages.error(request, "Please select a model to use.")
                return render(request, 'ml_app/bulk_test.html', {'trained_models': trained_models})
            
            try:
                model_record = TrainedModel.objects.get(id=model_id)
            except TrainedModel.DoesNotExist:
                messages.error(request, "Selected model not found.")
                return render(request, 'ml_app/bulk_test.html', {'trained_models': trained_models})
            
            # Save file temporarily
            file_path = default_storage.save(f'temp/{csv_file.name}', csv_file)
            full_path = default_storage.path(file_path)
            
            # Read CSV file
            df = pd.read_csv(full_path)
            
            # Load the model
            ml_engine = PersonalityMLEngine()
            model_data = ml_engine.load_model(model_record.model_file)
            
            # Make predictions
            predictions = []
            for _, row in df.iterrows():
                # Prepare input features (assuming same structure as training data)
                input_features = {}
                for col in df.columns:
                    if col != 'Personality':  # Skip target column if present
                        input_features[col] = row[col]
                
                # Make prediction
                prediction_result = ml_engine.predict_single(model_data, input_features)
                predictions.append({
                    'input': input_features,
                    'prediction': prediction_result['prediction'],
                    'confidence': prediction_result['confidence'],
                    'probabilities': prediction_result['probabilities']
                })
            
            # Calculate bulk statistics
            total_predictions = len(predictions)
            introvert_count = sum(1 for p in predictions if p['prediction'] == 'Introvert')
            extrovert_count = sum(1 for p in predictions if p['prediction'] == 'Extrovert')
            avg_confidence = sum(p['confidence'] for p in predictions) / total_predictions
            
            # Save results to database
            for i, pred in enumerate(predictions):
                PredictionResult.objects.create(
                    model_used=model_record,
                    input_data=pred['input'],
                    prediction=pred['prediction'],
                    confidence=pred['confidence'],
                    probabilities=pred['probabilities']
                )
            
            # Prepare results for display
            bulk_results = {
                'total_predictions': total_predictions,
                'introvert_count': introvert_count,
                'extrovert_count': extrovert_count,
                'avg_confidence': avg_confidence,
                'model_used': model_record.name,
                'predictions': predictions[:10],  # Show first 10 for display
                'all_predictions': predictions
            }
            
            messages.success(request, f"Successfully processed {total_predictions} predictions using {model_record.name}")
            
            return render(request, 'ml_app/bulk_test_results.html', {
                'bulk_results': bulk_results
            })
            
        except Exception as e:
            messages.error(request, f"Error processing bulk test: {str(e)}")
            return render(request, 'ml_app/bulk_test.html', {'trained_models': trained_models})
    
    return render(request, 'ml_app/bulk_test.html', {'trained_models': trained_models}) 