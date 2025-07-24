import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
from pathlib import Path
from django.conf import settings

class PersonalityMLEngine:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.feature_names = [
            'time_spent_alone', 'stage_fear', 'social_event_attendance',
            'going_outside', 'drained_after_socializing', 'friends_circle_size', 'post_frequency'
        ]
        self.categorical_features = ['stage_fear', 'drained_after_socializing']
        self.numerical_features = ['time_spent_alone', 'social_event_attendance', 'going_outside', 'friends_circle_size', 'post_frequency']
    
    def load_and_preprocess_data(self, csv_path):
        """Load and preprocess the dataset"""
        # Load data
        df = pd.read_csv(csv_path)
        
        # Rename columns to match our model
        column_mapping = {
            'Time_spent_Alone': 'time_spent_alone',
            'Stage_fear': 'stage_fear',
            'Social_event_attendance': 'social_event_attendance',
            'Going_outside': 'going_outside',
            'Drained_after_socializing': 'drained_after_socializing',
            'Friends_circle_size': 'friends_circle_size',
            'Post_frequency': 'post_frequency',
            'Personality': 'personality'
        }
        df = df.rename(columns=column_mapping)
        
        # Encode categorical features
        for feature in self.categorical_features:
            df[feature] = self.label_encoder.fit_transform(df[feature])
        
        # Encode target variable
        df['personality'] = self.label_encoder.fit_transform(df['personality'])
        
        return df
    
    def preprocess_data_from_df(self, df):
        """Preprocess data from DataFrame - automatically detect target as last column"""
        # Get the last column as target
        target_column = df.columns[-1]
        feature_columns = df.columns[:-1]  # All columns except the last one
        
        # Update feature names based on actual data
        self.feature_names = list(feature_columns)
        
        # Identify categorical and numerical features
        categorical_features = []
        numerical_features = []
        
        for col in self.feature_names:
            if df[col].dtype == 'object' or df[col].nunique() < 10:
                categorical_features.append(col)
            else:
                numerical_features.append(col)
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        
        # Encode categorical features
        for feature in self.categorical_features:
            df[feature] = self.label_encoder.fit_transform(df[feature])
        
        # Encode target variable
        df[target_column] = self.label_encoder.fit_transform(df[target_column])
        
        return df
    
    def prepare_features(self, df):
        """Prepare features for training"""
        # Get the target column (last column)
        target_column = df.columns[-1]
        
        # Separate features and target
        X = df[self.feature_names]
        y = df[target_column]
        
        # Scale numerical features
        X_scaled = X.copy()
        if self.numerical_features:
            X_scaled[self.numerical_features] = self.scaler.fit_transform(X[self.numerical_features])
        
        # Feature selection
        selector = SelectKBest(score_func=f_classif, k=min(5, len(self.feature_names)))
        X_selected = selector.fit_transform(X_scaled, y)
        
        # Store feature selector and selected feature names
        self.feature_selector = selector
        self.selected_feature_names = [self.feature_names[i] for i in range(len(self.feature_names)) if selector.get_support()[i]]
        
        return X_selected, y
    
    def train_logistic_regression(self, X, y, model_name):
        """Train Logistic Regression model"""
        print(f"\n=== Training {model_name} ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Input y value counts: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Training y value counts: {np.bincount(y_train)}")
        print(f"Test y value counts: {np.bincount(y_test)}")
        
        # Train model
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Prediction value counts: {np.bincount(y_pred)}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # Get feature importance (coefficients)
        feature_importance = dict(zip(self.feature_names, np.abs(model.coef_[0])))
        
        # Model parameters
        model_params = {
            'C': model.C,
            'max_iter': model.max_iter,
            'random_state': model.random_state
        }
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'model_params': model_params,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def train_random_forest(self, X, y, model_name):
        """Train Random Forest model"""
        print(f"\n=== Training {model_name} ===")
        print(f"Input X shape: {X.shape}")
        print(f"Input y shape: {y.shape}")
        print(f"Input y value counts: {np.bincount(y)}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        print(f"Training set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")
        print(f"Training y value counts: {np.bincount(y_train)}")
        print(f"Test y value counts: {np.bincount(y_test)}")
        
        # Train model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        print(f"Predictions shape: {y_pred.shape}")
        print(f"Prediction value counts: {np.bincount(y_pred)}")
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"Confusion Matrix:\n{cm}")
        
        # Get feature importance
        feature_importance = dict(zip(self.feature_names, model.feature_importances_))
        
        # Model parameters
        model_params = {
            'n_estimators': model.n_estimators,
            'max_depth': model.max_depth,
            'random_state': model.random_state
        }
        
        return {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'feature_importance': feature_importance,
            'model_params': model_params,
            'X_train': X_train,
            'y_train': y_train,
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
    
    def save_model(self, model_result, model_type, model_name):
        """Save trained model"""
        # Create models directory
        models_dir = Path(settings.MODEL_STORAGE_PATH)
        models_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model file
        model_file_path = models_dir / f"{model_name}.joblib"
        
        # Save all components
        model_data = {
            'model': model_result['model'],
            'scaler': self.scaler,
            'label_encoder': self.label_encoder,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names,
            'selected_feature_names': getattr(self, 'selected_feature_names', self.feature_names)
        }
        
        joblib.dump(model_data, model_file_path)
        
        return str(model_file_path)
    
    def load_model(self, model_path):
        """Load saved model"""
        return joblib.load(model_path)
    
    def predict_single(self, model_data, input_features):
        """Make prediction for a single input"""
        # Get preprocessing objects from loaded model
        scaler = model_data['scaler']
        label_encoder = model_data['label_encoder']
        feature_selector = model_data['feature_selector']
        feature_names = model_data['feature_names']
        selected_feature_names = model_data.get('selected_feature_names', feature_names)
        
        print(f"Input features: {input_features}")
        print(f"Feature names: {feature_names}")
        print(f"Selected feature names: {selected_feature_names}")
        
        # Create a DataFrame with the input features using the exact same column names as training
        input_df = pd.DataFrame([input_features])
        
        print(f"Input DataFrame columns: {input_df.columns.tolist()}")
        print(f"Input DataFrame values: {input_df.values}")
        
        # Ensure all required features are present with correct names
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0.0  # Default value for missing features
        
        # Select only the features that were used in training, in the correct order
        input_df = input_df[feature_names]
        
        print(f"Input DataFrame after feature selection: {input_df.columns.tolist()}")
        print(f"Input DataFrame values after selection: {input_df.values}")
        
        # Encode categorical features
        categorical_features = ['Stage_fear', 'Drained_after_socializing']
        for feature in categorical_features:
            if feature in input_df.columns:
                # Handle encoding safely
                try:
                    input_df[feature] = label_encoder.transform(input_df[feature])
                except:
                    # If encoding fails, use a default value
                    input_df[feature] = 0
        
        # Scale numerical features
        numerical_features = ['Time_spent_Alone', 'Social_event_attendance', 'Going_outside', 'Friends_circle_size', 'Post_frequency']
        available_numerical = [f for f in numerical_features if f in input_df.columns]
        
        # Create a copy of the input for scaling
        input_scaled = input_df.copy()
        
        # Apply scaling only to available numerical features
        if available_numerical:
            try:
                input_scaled[available_numerical] = scaler.transform(input_df[available_numerical])
            except Exception as e:
                print(f"Scaling error: {e}")
                # If scaling fails, use original values
                input_scaled[available_numerical] = input_df[available_numerical]
        
        print(f"Input scaled values: {input_scaled.values}")
        
        # Apply feature selection
        try:
            input_selected = feature_selector.transform(input_scaled)
        except Exception as e:
            print(f"Feature selection error: {e}")
            # If feature selection fails, use all features
            input_selected = input_scaled.values
        
        print(f"Input selected shape: {input_selected.shape}")
        print(f"Input selected values: {input_selected}")
        
        # Make prediction
        model = model_data['model']
        raw_prediction = model.predict(input_selected)
        prediction_probabilities = model.predict_proba(input_selected)
        
        print(f"Raw prediction: {raw_prediction}")
        print(f"Prediction probabilities: {prediction_probabilities}")
        
        # Decode prediction
        try:
            decoded_prediction = label_encoder.inverse_transform(raw_prediction)[0]
        except:
            # If decoding fails, use the raw prediction
            decoded_prediction = "Introvert" if raw_prediction[0] == 0 else "Extrovert"
        
        print(f"Decoded prediction label: {decoded_prediction}")
        
        # Calculate confidence
        confidence = max(prediction_probabilities[0]) * 100  # Convert to percentage
        
        # Create probabilities dictionary
        probabilities = {}
        try:
            unique_labels = label_encoder.classes_
            for i, label in enumerate(unique_labels):
                probabilities[label] = prediction_probabilities[0][i] * 100  # Convert to percentage
        except:
            # Fallback probabilities
            probabilities = {'Introvert': prediction_probabilities[0][0] * 100, 'Extrovert': prediction_probabilities[0][1] * 100}
        
        print(f"Final prediction label: {decoded_prediction}")
        print(f"Confidence: {confidence}")
        print(f"Probabilities: {probabilities}")
        
        return decoded_prediction, confidence, probabilities
    
    def create_performance_plots(self, model_result, model_name):
        """Create performance visualization plots"""
        # Confusion Matrix
        cm = confusion_matrix(model_result['y_test'], model_result['y_pred'])
        
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted Introvert', 'Predicted Extrovert'],
            y=['Actual Introvert', 'Actual Extrovert'],
            colorscale='Blues',
            text=cm,
            texttemplate='%{text}',
            textfont={"size": 16}
        ))
        
        fig_cm.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted',
            yaxis_title='Actual'
        )
        
        # Feature Importance
        feature_importance = model_result['feature_importance']
        features = list(feature_importance.keys())
        importance_values = list(feature_importance.values())
        
        fig_importance = go.Figure(data=go.Bar(
            x=importance_values,
            y=features,
            orientation='h',
            marker_color='lightblue'
        ))
        
        fig_importance.update_layout(
            title=f'Feature Importance - {model_name}',
            xaxis_title='Importance',
            yaxis_title='Features'
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(model_result['y_test'], model_result['y_pred_proba'][:, 1])
        roc_auc = auc(fpr, tpr)
        
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC curve (AUC = {roc_auc:.3f})'
        ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random',
            line=dict(dash='dash')
        ))
        
        fig_roc.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate'
        )
        
        return fig_cm, fig_importance, fig_roc
    
    def get_model_summary(self, model_result, model_name):
        """Get model performance summary"""
        return {
            'model_name': model_name,
            'accuracy': model_result['accuracy'],
            'precision': model_result['precision'],
            'recall': model_result['recall'],
            'f1_score': model_result['f1_score'],
            'feature_importance': model_result['feature_importance']
        } 