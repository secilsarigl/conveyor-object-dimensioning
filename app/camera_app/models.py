from django.db import models

class Measurement(models.Model):
    product_id = models.CharField(max_length=100)
    product_name = models.CharField(max_length=100)
    camera1_width = models.FloatField()
    camera1_height = models.FloatField()
    camera2_width = models.FloatField()
    camera2_height = models.FloatField()
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.product_name} - {self.product_id}"
