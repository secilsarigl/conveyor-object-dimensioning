from django.db.models.signals import post_migrate
from django.dispatch import receiver
from .models import Measurement

@receiver(post_migrate)
def clear_measurements(sender, **kwargs):
    Measurement.objects.all().delete()
