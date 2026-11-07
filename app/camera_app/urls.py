# camera_app/urls.py
from django.urls import path
from .views import process_cameras, welcome, register
from django.contrib.auth.views import LoginView

urlpatterns = [
    path('', welcome, name='welcome'),
    path('process_cameras/', process_cameras, name='process_cameras'),
    path('register/', register, name='register'),
    path('login/', LoginView.as_view(template_name='camera_app/login.html'), name='login'),
]
