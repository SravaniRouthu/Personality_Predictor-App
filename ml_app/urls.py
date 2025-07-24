from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload-data/', views.upload_data, name='upload_data'),
    path('train-models/', views.train_models, name='train_models'),
    path('predict/', views.predict_personality, name='predict_personality'),
    path('bulk-test/', views.bulk_test, name='bulk_test'),
    path('data-exploration/', views.data_exploration, name='data_exploration'),
    path('about/', views.about, name='about'),
    path('reset/', views.reset_application, name='reset_application'),
] 