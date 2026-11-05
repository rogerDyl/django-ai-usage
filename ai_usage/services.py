import time

import litellm
from agents import Agent, Runner
from agents.exceptions import AgentsException
from agents.extensions.models.litellm_model import LitellmModel

from ai_usage.models import AIUsage


class AIClient:
    def __init__(self, feature):
        self.feature = feature

    def run(self, model, instructions, message, max_turns=1, **agent_kwargs):
        agent = Agent(
            name=self.feature,
            model=LitellmModel(model=model),
            instructions=instructions,
            **agent_kwargs
        )
        started_at = time.monotonic()
        try:
            result = Runner.run_sync(agent, message, max_turns=max_turns)
        except AgentsException as exception:
            # l'agent a échoué en cours de route : les tours déjà faits ont été payés
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
            # modèle absent de la table de prix de litellm : le coût reste vide
            return None
        return input_cost + output_cost
