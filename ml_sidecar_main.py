import asyncio
import logging
import sys
from collections.abc import Callable
from typing import Any

from pi_sidecar.ipc.ndjson_transport import NdjsonTransport
from pi_sidecar.inference.engine import InferenceEngine
from pi_sidecar.models.sidecar_registry import ModelRegistry
from pi_sidecar.training import TrainingService

logging.basicConfig(
    stream=sys.stderr,
    level=logging.INFO,
    format="[ml-sidecar] %(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


class MlRequestHandler:
    def __init__(self, engine: InferenceEngine, registry: ModelRegistry):
        self.engine = engine
        self.registry = registry
        self.training = TrainingService()
        self._handlers = {
            "health.ping": self._health_ping,
            "lifecycle.shutdown": self._lifecycle_shutdown,
            "inference.complete": self._inference_complete,
            "inference.embed": self._inference_embed,
            "inference.plan": self._inference_plan,
            "inference.load_model": self._inference_load_model,
            "model.list": self._model_list,
            "model.load": self._model_load,
            "model.unload": self._model_unload,
            "model.download": self._model_download,
            "training.start": self._training_start,
            "training.stop": self._training_stop,
            "training.status": self._training_status,
            "training.list": self._training_list,
            "voice.synthesize": self._voice_synthesize,
            "voice.transcribe": self._voice_transcribe,
            "personality.hatch_chat": self._personality_hatch_chat,
        }

    async def dispatch(
        self, method: str, params: dict, progress_callback: Callable | None = None
    ) -> Any:
        handler = self._handlers.get(method)
        if not handler:
            raise ValueError(f"Method {method} not supported by ML sidecar")
        return await handler(params, progress_callback)

    async def _health_ping(self, p, _cb):
        return {"status": "ok", "sidecar": "ml"}

    async def _lifecycle_shutdown(self, p, _cb):
        asyncio.get_event_loop().call_later(0.5, sys.exit, 0)
        return {"status": "shutting_down"}

    async def _inference_complete(self, p, cb):
        if p.get("stream"):
            text = ""
            async for token in self.engine.complete_stream(
                **{k: v for k, v in p.items() if k != "stream"}
            ):
                text += token
                if cb:
                    await cb({"token": token, "is_streaming": True})
            return {"text": text}
        else:
            return await self.engine.complete(**p)

    async def _inference_embed(self, p, _cb):
        vector = await self.engine.embed(
            text=p["text"], model_id=p.get("model_id", "all-MiniLM-L6-v2")
        )
        return {"embedding": vector}

    async def _inference_plan(self, p, _cb):
        return await self.engine.plan(**p)

    async def _inference_load_model(self, p, _cb):
        return await self.engine.load_model(p.get("model_id") or p.get("path"))

    async def _model_list(self, p, _cb):
        return {"models": self.registry.list_models()}

    async def _model_load(self, p, _cb):
        await self.registry.load_model(p["model_id"])
        return {"status": "loaded", "model_id": p["model_id"]}

    async def _model_unload(self, p, _cb):
        unloaded = self.registry.unload_model(p["model_id"])
        return {"status": "unloaded" if unloaded else "not_loaded", "model_id": p["model_id"]}

    async def _model_download(self, p, cb):
        result = await self.registry.download_model(p["model_id"], progress_callback=cb)
        return result

    async def _training_start(self, p, _cb):
        return {"run_id": await self.training.start(p), "status": "started"}

    async def _training_stop(self, p, _cb):
        return {"stopped": await self.training.stop(p["run_id"]), "run_id": p["run_id"]}

    async def _training_status(self, p, _cb):
        return await self.training.status(p["run_id"])

    async def _training_list(self, p, _cb):
        return {"runs": await self.training.list_runs()}

    async def _voice_synthesize(self, p, _cb):
        from pi_sidecar.tts.elevenlabs import ElevenLabsTTS

        tts = ElevenLabsTTS(api_key=p.get("api_key"))
        success = await tts.synthesize(p["text"], p["output_path"])
        return {"success": success, "output_path": p["output_path"]}

    async def _voice_transcribe(self, p, _cb):
        from pi_sidecar.stt.whisper import WhisperSTT

        stt = WhisperSTT(model_size=p.get("model_size", "base"), device=p.get("device", "cpu"))
        text = await stt.transcribe(p["audio_path"])
        return {"text": text, "audio_path": p["audio_path"]}

    async def _personality_hatch_chat(self, p, _cb):
        from pi_sidecar.personality import get_personality

        personality = get_personality()
        system_prompt = f"{personality.system_prompt}\n\n# Hatching Context\nYou are in the 'hatching' phase. Be extremely welcoming and discuss your identity with the user."
        history = p.get("history", [])
        prompt = f"{system_prompt}\n\n"
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            prompt += f"[{role}]: {content}\n"
        if p.get("stream"):
            text = ""
            async for token in self.engine.complete_stream(
                prompt=prompt,
                provider=p.get("provider", "local"),
                model_id=p.get("model_id"),
            ):
                text += token
                if _cb:
                    await _cb({"token": token, "is_streaming": True})
            return {"text": text}
        else:
            result = await self.engine.complete(
                prompt=prompt,
                provider=p.get("provider", "local"),
                model_id=p.get("model_id"),
            )
            return {"text": result.get("text", "")}


async def main():
    logger.info("ML sidecar starting")
    registry = ModelRegistry()
    engine = InferenceEngine(registry)
    handler = MlRequestHandler(engine, registry)
    transport = NdjsonTransport()
    async for request in transport.read_requests():
        asyncio.create_task(_handle_request(handler, transport, request))


async def _handle_request(handler, transport, request):
    req_id = request.get("id", "unknown")
    try:
        result = await handler.dispatch(
            request["method"],
            request.get("params", {}),
            lambda p: asyncio.ensure_future(transport.send_progress(req_id, p)),
        )
        await transport.send_response(req_id, result=result)
    except Exception as e:
        logger.exception("Error handling %s", request["method"])
        await transport.send_error(req_id, code=type(e).__name__, message=str(e))


if __name__ == "__main__":
    asyncio.run(main())
