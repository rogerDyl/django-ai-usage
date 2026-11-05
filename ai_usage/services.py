import base64
import mimetypes
import time
from logging import getLogger
from typing import IO, Any

import litellm
from agents import Agent, ModelSettings, RunConfig, Runner
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
        files: list[tuple[str, IO[bytes]] | tuple[str, IO[bytes], str]] | None = None,
        timeout: float = 600,
        model_settings: ModelSettings | None = None,
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
            model_settings=(model_settings or ModelSettings()).resolve(
                ModelSettings(extra_args={"timeout": timeout})
            ),
            **agent_kwargs,
        )
        file_inputs = []
        try:
            file_inputs = self._build_file_inputs(
                provider, api_key, files or [], timeout
            )
            message = self._add_files(message, file_inputs)
            return self._run_agent(agent, model, message, max_turns, tracing_disabled)
        finally:
            self._delete_files(provider, api_key, file_inputs)

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

    def _upload_file(self, provider, api_key, filename, content, timeout):
        uploaded_file = litellm.create_file(
            file=(filename, content),
            # litellm's Literal omits "user_data", which OpenAI requires for input_file
            purpose="user_data",  # type: ignore[arg-type]
            custom_llm_provider=provider,
            api_key=api_key,
            timeout=timeout,
        )
        return uploaded_file.id

    def _build_file_inputs(self, provider, api_key, files, timeout):
        file_inputs = []
        for filename, content, *optional_mime_type in files:
            mime_type = self._get_mime_type(filename, optional_mime_type)
            if mime_type and mime_type.startswith("image/"):
                encoded_content = base64.b64encode(content.read()).decode("ascii")
                file_inputs.append(
                    {
                        "type": "input_image",
                        "image_url": f"data:{mime_type};base64,{encoded_content}",
                    }
                )
            else:
                file_id = self._upload_file(
                    provider, api_key, filename, content, timeout
                )
                file_inputs.append({"type": "input_file", "file_id": file_id})
        return file_inputs

    def _get_mime_type(self, filename, optional_mime_type):
        if optional_mime_type:
            return optional_mime_type[0]
        else:
            return mimetypes.guess_type(filename)[0]

    def _add_files(self, message, file_inputs):
        if not file_inputs:
            return message

        if isinstance(message, str):
            return [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": message}, *file_inputs],
                }
            ]
        else:
            return [*message, {"role": "user", "content": file_inputs}]

    def _delete_files(self, provider, api_key, file_inputs):
        for file_input in file_inputs:
            file_id = file_input.get("file_id")
            if file_id:
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
