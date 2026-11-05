import time
from logging import getLogger
from typing import IO, Any

import litellm
from agents import Agent, RunConfig, Runner
from openai.types.responses import ResponseInputItemParam
from agents.exceptions import AgentsException
from agents.extensions.models.litellm_model import LitellmModel
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from ai_usage.models import AIUsage

logger = getLogger(__name__)


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
        files: list[tuple[str, IO[bytes]]] | None = None,
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
        uploaded_file_ids = []
        try:
            for filename, content in files or []:
                uploaded_file_ids.append(
                    self._upload_file(provider, api_key, filename, content)
                )
            message = self._add_files(message, uploaded_file_ids)
            return self._run_agent(agent, model, message, max_turns, tracing_disabled)
        finally:
            self._delete_files(provider, api_key, uploaded_file_ids)

    def _run_agent(self, agent, model, message, max_turns, tracing_disabled):
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

    def _upload_file(self, provider, api_key, filename, content):
        uploaded_file = litellm.create_file(
            file=(filename, content),
            purpose="user_data",
            custom_llm_provider=provider,
            api_key=api_key,
        )
        return uploaded_file.id

    def _add_files(self, message, file_ids):
        if not file_ids:
            return message

        file_inputs = [
            {"type": "input_file", "file_id": file_id} for file_id in file_ids
        ]
        if isinstance(message, str):
            return [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}, *file_inputs],
                }
            ]
        else:
            return [*message, {"role": "user", "content": file_inputs}]

    def _delete_files(self, provider, api_key, file_ids):
        for file_id in file_ids:
            try:
                litellm.file_delete(
                    file_id, custom_llm_provider=provider, api_key=api_key
                )
            except Exception:
                logger.warning(f"Could not delete the uploaded file {file_id}")

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
