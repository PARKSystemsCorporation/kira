from __future__ import annotations

import uuid
from typing import Any, Optional


class Kira:
    """
    Public API facade for KIRA core.

    Parameters
    ----------
    backend:
        Backend name (default: "ollama").
    model:
        Model identifier for the backend (default: "llama3.1:8b").
    memory_path:
        Path to local memory store (default: "~/.kira/memory.db").
    decay_mode:
        Memory decay mode: "linear", "exponential", "usage_based", or "none".
    decay_half_life_days:
        Days for importance to halve in exponential mode.
    decay_min_importance:
        Minimum importance floor after decay.
    reinforcement_mode:
        Reinforcement strategy (default: "correlation").
    reinforcement_schedule:
        Reinforcement schedule: "on_use", "daily", "hourly", or "manual".
    reinforcement_strength:
        Multiplier for reinforcement application.
    correlation_threshold:
        Similarity threshold for correlation-based reinforcement.
    verbose:
        Enable verbose logging and diagnostics.
    **kwargs:
        Reserved for future configuration.
    """

    def __init__(
        self,
        backend: str = "ollama",
        model: str = "llama3.1:8b",
        memory_path: str = "~/.kira/memory.db",
        decay_mode: str = "exponential",
        decay_half_life_days: float = 30.0,
        decay_min_importance: float = 0.05,
        reinforcement_mode: str = "correlation",
        reinforcement_schedule: str = "on_use",
        reinforcement_strength: float = 1.2,
        correlation_threshold: float = 0.7,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        self._config: dict[str, Any] = {
            "backend": backend,
            "model": model,
            "memory_path": memory_path,
            "decay_mode": decay_mode,
            "decay_half_life_days": decay_half_life_days,
            "decay_min_importance": decay_min_importance,
            "reinforcement_mode": reinforcement_mode,
            "reinforcement_schedule": reinforcement_schedule,
            "reinforcement_strength": reinforcement_strength,
            "correlation_threshold": correlation_threshold,
            "verbose": verbose,
            **kwargs,
        }

        self._memory = None
        self._router = None
        self._backend = None

    def _ensure_internals(self) -> None:
        if self._memory is None:
            from ._internals.memory import MemoryOrchestrator

            self._memory = MemoryOrchestrator(self._config)

        if self._router is None:
            from ._internals.router import Router

            self._router = Router(self._config)

    def _call_backend(self, prompt: str) -> str:
        backend = self._config.get("backend", "ollama")
        if backend != "ollama":
            raise NotImplementedError(f"Backend '{backend}' is not supported yet.")

        try:
            import ollama
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("Ollama backend requested but 'ollama' is not installed.") from exc

        model = self._config.get("model", "llama3.1:8b")
        response = ollama.chat(model=model, messages=[{"role": "user", "content": prompt}])
        if isinstance(response, dict):
            message = response.get("message") or {}
            return message.get("content", "")
        return str(response)

    def _reinforce_if_needed(self) -> None:
        if self._memory is None:
            return
        self._memory.reinforce_if_needed()

    def chat(self, message: str, system: Optional[str] = None) -> str:
        """
        Send a message through KIRA and return the model response.

        Parameters
        ----------
        message:
            User message content.
        system:
            Optional system prompt override.
        """
        self._ensure_internals()

        memory_context = self._memory.get_memory_context()
        full_prompt = self._router.build_prompt(message, memory_context, system)
        response = self._call_backend(full_prompt)

        self._memory.process_message(message, message_id=str(uuid.uuid4()), user_id="local")
        self._memory.store_interaction(message, response)
        self._reinforce_if_needed()

        return response