import yaml
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Singleton class to securely load and serve YAML configurations.
    Prevents hardcoding prompts and model IDs inside core logic.
    """
    _instance = None
    _config = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # Resolve the absolute path to the configs/system_config.yaml file
        base_dir = Path(__file__).resolve().parent.parent.parent
        config_path = base_dir / "configs" / "system_config.yaml"
        
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info("✅ [ConfigManager] System configuration loaded successfully.")
        except Exception as e:
            logger.critical(f"❌ [ConfigManager] Failed to load config file: {e}")
            raise

    def get_models(self):
        return self._config.get("models", {})

    def get_prompt(self, prompt_name: str) -> str:
        return self._config.get("prompts", {}).get(prompt_name, "")
        
# Global instance to be imported by other modules
config = ConfigManager()
