import json
import os
from typing import Any, Dict


class ConfigManager:
    def __init__(self, path: str = "app_config.json") -> None:
        self.path = path
        self._data: Dict[str, Any] = {
            "confidence_threshold": 70,
            "remove_urls": True,
            "remove_mentions": True,
            "model": "cardiffnlp/twitter-roberta-base-sentiment",
        }
        self._load()

    def _load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    file_data = json.load(f)
                if isinstance(file_data, dict):
                    self._data.update(file_data)
        except Exception:
            # If loading fails, keep defaults
            pass

    def _save(self) -> None:
        try:
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2)
        except Exception:
            # Ignore persistence errors to avoid breaking UI
            pass

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._data[key] = value
        self._save()

