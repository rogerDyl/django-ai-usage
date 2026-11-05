from django.db import models


class AIUsage(models.Model):
    provider = models.CharField(max_length=50)
    model = models.CharField(max_length=100)
    feature = models.CharField(max_length=100)
    input_tokens = models.PositiveIntegerField()
    output_tokens = models.PositiveIntegerField()
    turns = models.PositiveIntegerField()
    cost = models.DecimalField(max_digits=12, decimal_places=6, null=True)
    duration_ms = models.PositiveIntegerField()
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "AI usage"
        verbose_name_plural = "AI usages"

    def __str__(self):
        return f"{self.feature} - {self.model} - {self.created_at}"
