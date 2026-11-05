from django.db import models


class AIUsage(models.Model):
    provider = models.CharField(max_length=50)
    model = models.CharField(max_length=100)
    feature = models.CharField(max_length=100)
    reasoning_effort = models.CharField(max_length=20, blank=True)
    input_tokens = models.PositiveIntegerField()
    cached_tokens = models.PositiveIntegerField(default=0)
    cache_write_tokens = models.PositiveIntegerField(default=0)
    output_tokens = models.PositiveIntegerField()
    reasoning_tokens = models.PositiveIntegerField(default=0)
    total_tokens = models.PositiveIntegerField(default=0)
    turns = models.PositiveIntegerField()
    cost = models.DecimalField(max_digits=12, decimal_places=6, null=True)
    duration_ms = models.PositiveIntegerField()
    succeeded = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        verbose_name = "AI usage"
        verbose_name_plural = "AI usages"

    def __str__(self):
        return f"{self.feature} - {self.model} - {self.created_at}"
