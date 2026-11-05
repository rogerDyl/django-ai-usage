import time
from typing import Any

import litellm
from agents import Agent, RunConfig, Runner
from openai.types.responses import ResponseInputItemParam
from agents.exceptions import AgentsException
from agents.extensions.models.litellm_model import LitellmModel
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from ai_usage.models import AIUsage


class AIClient:
    def __init__(self, feature: str):
        self.feature = feature

    def run(
        self,
        model: str,
        instructions: str,
        message: str | list[ResponseInputItemParam],
        max_turns: int = 1,
        tracing_disabled: bool = False,
        **agent_kwargs: Any,
    ) -> Any:
        provider = model.split("/", 1)[0]
        api_key = settings.AI_USAGE_API_KEYS.get(provider)
        if not api_key:
            raise ImproperlyConfigured(
                f"Missing API key for provider '{provider}' in AI_USAGE_API_KEYS"
            )

        agent = Agent(
            name=self.feature,
            model=LitellmModel(model=model, api_key=api_key),
            instructions=instructions,
            **agent_kwargs,
        )
        started_at = time.monotonic()
        try:
            result = Runner.run_sync(
                agent,
                message,
                max_turns=max_turns,
                run_config=RunConfig(tracing_disabled=tracing_disabled),
            )
        except AgentsException as exception:
            # The agent failed midway: the turns already run have been paid for
            if exception.run_data is not None:
                self._store_usage(
                    model, exception.run_data.context_wrapper.usage, started_at
                )
            raise
        self._store_usage(model, result.context_wrapper.usage, started_at)
        return result.final_output

    def _store_usage(self, model, usage, started_at):
        if usage.requests == 0:
            return

        provider, model_name = model.split("/", 1)
        AIUsage.objects.create(
            provider=provider,
            model=model_name,
            feature=self.feature,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            turns=usage.requests,
            cost=self._compute_cost(model, usage),
            duration_ms=(time.monotonic() - started_at) * 1000,
        )

    def _compute_cost(self, model, usage):
        try:
            input_cost, output_cost = litellm.cost_per_token(
                model=model,
                prompt_tokens=usage.input_tokens,
                completion_tokens=usage.output_tokens,
                cache_read_input_tokens=usage.input_tokens_details.cached_tokens,
            )
        except Exception:
            # Model missing from the litellm price map: the cost stays empty
            return None
        return input_cost + output_cost
