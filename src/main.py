import hydra
from omegaconf import DictConfig
from loguru import logger
import torch

from asr import transcribe_audio

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    device = torch.device("mps") if torch.backends.mps.is_available() \
             else torch.device("cuda") if torch.cuda.is_available() \
             else torch.device("cpu")
    logger.info(f"Using device: {device}")

    transcribed_text = transcribe_audio(cfg.asr, device)
    
    logger.info(f"Transcribed Text: {transcribed_text}")

if __name__ == "__main__":
    main()