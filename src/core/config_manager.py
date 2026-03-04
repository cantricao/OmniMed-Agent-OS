import os
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
        # [ENTERPRISE FIX] 1. Allow using Environment Variables to override path
        env_config_path = os.getenv("OMNIMED_CONFIG_PATH")

        if env_config_path and Path(env_config_path).exists():
            config_path = Path(env_config_path)
        else:
            # [ENTERPRISE FIX] 2. Automatically traverse upwards to find the root config directory
            current_dir = Path(__file__).resolve().parent
            config_path = None

            # Traverse up to the parent directory until reaching the drive root
            while current_dir.name != "" and current_dir.name != "/":
                potential_path = current_dir / "configs" / "system_config.yaml"
                if potential_path.exists():
                    config_path = potential_path
                    break
                current_dir = current_dir.parent

            if config_path is None:
                raise FileNotFoundError(
                    "CRITICAL: Could not locate 'configs/system_config.yaml' in any parent directory."
                )

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                self._config = yaml.safe_load(f)
            logger.info(
                f"✅ [ConfigManager] System configuration loaded from: {config_path}"
            )
        except Exception as e:
            logger.critical(f"❌ [ConfigManager] Failed to load config file: {e}")
            raise

    def get_models(self):
        return self._config.get("models", {})

    def get_prompt(self, prompt_name: str) -> str:
        return self._config.get("prompts", {}).get(prompt_name, "")


# Global instance to be imported by other modules
config = ConfigManager()
