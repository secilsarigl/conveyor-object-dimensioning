from django.apps import AppConfig

class CameraAppConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'camera_app'

    def ready(self):
        import camera_app.signals

