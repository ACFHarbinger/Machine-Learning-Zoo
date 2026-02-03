"""vLLM optimized inference engine."""

import logging
from typing import Any, Dict, List, Union

logger = logging.getLogger(__name__)


class VLLMEngine:
    """
    Inference engine using vLLM for high-throughput serving of LLMs.
    """

    def __init__(
        self,
        model_name: str,
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """
        Initialize the vLLM engine.
        """
        self.model_name = model_name
        self.engine = None

        try:
            from vllm import LLM, SamplingParams

            self.LLM = LLM
            self.SamplingParams = SamplingParams

            logger.info(f"Initializing vLLM engine for model: {model_name}")
            self.engine = self.LLM(
                model=model_name,
                tensor_parallel_size=tensor_parallel_size,
                gpu_memory_utilization=gpu_memory_utilization,
                trust_remote_code=trust_remote_code,
                **kwargs,
            )
        except ImportError:
            logger.warning("vLLM not installed. VLLMEngine will operate in mock mode.")
            self.engine = None

    def generate(
        self,
        prompts: Union[str, List[str]],
        max_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        **kwargs,
    ) -> List[str]:
        """
        Generate text using vLLM.
        """
        if isinstance(prompts, str):
            prompts = [prompts]

        if self.engine is None:
            logger.warning("vLLM engine not initialized. Returning mock response.")
            return [f"[Mock vLLM Response for: {p[:20]}...]" for p in prompts]

        sampling_params = self.SamplingParams(
            max_tokens=max_tokens, temperature=temperature, top_p=top_p, **kwargs
        )

        outputs = self.engine.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            results.append(output.outputs[0].text)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get engine statistics (if available).
        """
        if self.engine:
            # vLLM doesn't expose a simple stats dict easily without the server mode,
            # but we can provide basic info.
            return {"model": self.model_name, "engine": "vLLM", "status": "active"}
        return {"status": "mock"}
