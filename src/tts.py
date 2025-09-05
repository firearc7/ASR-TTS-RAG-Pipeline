from transformers import pipeline
from loguru import logger
import torch
import soundfile as sf
from datasets import load_dataset
import numpy as np

def text_to_speech(text: str, cfg: dict, device: torch.device):
    logger.info("Starting Text-to-Speech")
    
    embeddings_dataset = load_dataset(cfg.speaker_embeddings_model, split="validation")
    speaker_embeddings = torch.tensor(embeddings_dataset[cfg.speaker_index]["xvector"]).unsqueeze(0)
    speaker_embeddings = speaker_embeddings.to(device)

    tkn = cfg.hf_token if "hf_token" in cfg else None
    pipe = pipeline(
        "text-to-speech",
        model=cfg.model_name,
        device=device,
        token=tkn
    )
    
    output = pipe(text, forward_params={"speaker_embeddings": speaker_embeddings})

    sf.write(cfg.output_path, output["audio"], samplerate=output["sampling_rate"])
    logger.info(f"TTS output saved to {cfg.output_path}")
