# Roadmap — Speech (STT/TTS) & Personas

Audio pipelines and the Agent/Persona layer. Implementation in [`src/stt/`](../../src/stt/)
(`whisper`), [`src/tts/`](../../src/tts/) (`elevenlabs`), and `personality.py`. Feature overview in
[`docs/AGENTS.md`](../../docs/AGENTS.md).

## §1 — STT / TTS

- [ ] Document the Whisper STT and ElevenLabs TTS adapters (config, latency, offline vs API).
- [ ] Guard optional/heavy or network deps with actionable install messages; keys from env only.

## §2 — Personas

- [ ] Document the `Personality` layer and how `set_persona`/`chat` flow through the sidecar.
- [ ] Persona configs live under `src/configs/` (Hydra); no hard-coded persona strings.

## §3 — Tests

- [ ] Adapter tests mock the network; persona selection covered deterministically.
