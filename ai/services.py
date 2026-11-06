import base64
import mimetypes
import time
from logging import getLogger
from typing import IO, Any

import litellm
from agents import Agent, ModelSettings, RunConfig, Runner
from agents import Usage as AgentUsage
from openai.types.responses import ResponseInputItemParam
from agents.extensions.models.litellm_model import LitellmModel
from agents.tracing import TracingConfig
from litellm.decisions.call import OpenAIDecisionQuestions
from litellm.types.decisions import OpenAIDecisionAnswer, OpenAIDecisionInput
from django.conf import settings
from django.core.exceptions import ImproperlyConfigured

from ai.models import Usage

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
            model_settings=self._get_model_settings(
                provider, timeout, max_turns, model_settings
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

    def decide(
        self,
        model: str,
        input: OpenAIDecisionInput,
        questions: OpenAIDecisionQuestions,
        timeout: float = 600,
        **kwargs: Any,
    ) -> tuple[OpenAIDecisionAnswer, ...]:
        if "state" in kwargs:
            raise TypeError("decide() does not support the deprecated state= format")

        provider = model.split("/", 1)[0]
        api_key = settings.AI_USAGE_API_KEYS.get(provider)
        if not api_key:
            raise ImproperlyConfigured(
                f"Missing API key for provider '{provider}' in AI_USAGE_API_KEYS"
            )

        started_at = time.monotonic()
        try:
            response = litellm.decisions(
                model=model,
                input=input,
                questions=questions,
                api_key=api_key,
                timeout=timeout,
                **kwargs,
            )
        except Exception:
            self._store_usage(
                model, AgentUsage(), started_at, "decisions", succeeded=False
            )
            raise
        usage = AgentUsage(requests=1, **response.usage.model_dump())
        self._store_usage(model, usage, started_at, "decisions")
        return response.answers

    def _get_model_settings(
        self,
        provider: str,
        timeout: float,
        max_turns: int,
        model_settings: ModelSettings | None,
    ) -> ModelSettings:
        extra_args: dict[str, Any] = {"timeout": timeout}
        if provider == "anthropic" and max_turns > 1:
            extra_args["cache_control_injection_points"] = [
                {"location": "message", "index": -1}
            ]

        return ModelSettings(extra_args=extra_args).resolve(model_settings)

    def _run_agent(self, agent, model, message, max_turns, tracing_disabled):
        reasoning = agent.model_settings.reasoning
        if reasoning and reasoning.effort:
            reasoning_effort = reasoning.effort
        else:
            reasoning_effort = ""
        started_at = time.monotonic()
        try:
            result = Runner.run_sync(
                agent,
                message,
                max_turns=max_turns,
                run_config=self._configure_tracing(model, tracing_disabled),
            )
        except Exception as exception:
            # The agent failed midway: the turns already run have been paid for
            run_data = getattr(exception, "run_data", None)
            usage = run_data.context_wrapper.usage if run_data else AgentUsage()
            self._store_usage(
                model, usage, started_at, "agent", reasoning_effort, succeeded=False
            )
            raise
        self._store_usage(
            model, result.context_wrapper.usage, started_at, "agent", reasoning_effort
        )
        return result.final_output

    def _configure_tracing(self, model, tracing_disabled):
        provider = model.split("/", 1)[0]
        if provider == "openai" and not tracing_disabled:
            return RunConfig(
                tracing=TracingConfig(api_key=settings.AI_USAGE_API_KEYS[provider])
            )
        else:
            return RunConfig(tracing_disabled=True)

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

    def _store_usage(
        self,
        model,
        usage,
        started_at,
        call_type,
        reasoning_effort="",
        succeeded=True,
    ):
        if usage.requests == 0 and succeeded:
            return

        provider, model_name = model.split("/", 1)
        Usage.objects.create(
            provider=provider,
            model=model_name,
            feature=self.feature,
            call_type=call_type,
            reasoning_effort=reasoning_effort,
            input_tokens=usage.input_tokens,
            cached_tokens=usage.input_tokens_details.cached_tokens,
            cache_write_tokens=usage.input_tokens_details.cache_write_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.output_tokens_details.reasoning_tokens,
            total_tokens=usage.total_tokens,
            turns=usage.requests,
            cost=self._compute_cost(model, usage, call_type),
            duration_ms=(time.monotonic() - started_at) * 1000,
            succeeded=succeeded,
        )

    def _compute_cost(self, model, usage, call_type):
        cost = 0
        for request_usage in usage.request_usage_entries or [usage]:
            input_details = request_usage.input_tokens_details
            try:
                input_cost, output_cost = litellm.cost_per_token(
                    model=model,
                    prompt_tokens=request_usage.input_tokens,
                    completion_tokens=request_usage.output_tokens,
                    cache_read_input_tokens=input_details.cached_tokens,
                    cache_creation_input_tokens=input_details.cache_write_tokens,
                    call_type="decisions" if call_type == "decisions" else "completion",
                )
            except Exception:
                # Model missing from the litellm price map: the cost stays empty
                return None
            cost += input_cost + output_cost
        return cost
