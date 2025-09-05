import hydra
from omegaconf import DictConfig, OmegaConf
from loguru import logger

@hydra.main(config_path="../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entry point for the ASR-RAG-TTS pipeline.
    """
    logger.info(f"Starting {cfg.project_name}")
    logger.info("Configuration:\n" + OmegaConf.to_yaml(cfg))
    logger.info("Step 1: Project Scaffolding Complete!")

if __name__ == "__main__":
    main()