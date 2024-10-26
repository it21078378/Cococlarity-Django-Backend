from django.urls import path
from . import views

urlpatterns = [
    path('predict/', views.predict, name='predict'),
    path('google-trends/', views.google_trends, name='google_trends'),
]