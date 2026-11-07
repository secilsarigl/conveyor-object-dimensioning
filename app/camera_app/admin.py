from django.contrib import admin
from .models import Measurement

@admin.register(Measurement)
class MeasurementAdmin(admin.ModelAdmin):
    list_display = ('product_id', 'product_name', 'camera1_width', 'camera1_height', 'camera2_width', 'camera2_height', 'timestamp')
