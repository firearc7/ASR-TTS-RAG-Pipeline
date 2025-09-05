import torch
import whisper
from loguru import logger
import os

def transcribe_audio(cfg: dict, device: torch.device) -> str:
    logger.info(f"Starting ASR")

    model = whisper.load_model(cfg.model_name)

    if not os.path.exists(cfg.audio_path):
        logger.error(f"Audio file not found at: {cfg.audio_path}.")
        raise FileNotFoundError(f"Audio file not found: {cfg.audio_path}")

    logger.info(f"Transcribing audio from: {cfg.audio_path}")
    result = model.transcribe(cfg.audio_path)
    transcribed_text = result["text"]

    return transcribed_text