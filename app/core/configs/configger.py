from pathlib import Path
from typing import Any, Dict
import yaml


class PipelineConfigger:
    """Configuratore centrale unico per l'intero progetto OMIRA-q.
    Carica config.yml, i dataset in configs/data/ e i modelli in configs/model/.
    """

    def __init__(self, config_root: str | Path = "app/core/configs"):
        self.config_root = Path(config_root)
        self.global_config_path = self.config_root / "config.yml"

        self.global_cfg: Dict[str, Any] = {}
        self.data_configs: Dict[str, Any] = {}
        self.model_configs: Dict[str, Any] = {}
        self.backend_configs: Dict[str, Any] = {}

        self._load_all()

    def _load_yaml(self, path: Path) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"File di configurazione non trovato: {path}")
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def _load_all(self) -> None:
        # carica configurazione globale
        self.global_cfg = self._load_yaml(self.global_config_path)

        # carica configurazioni dataset in core/configs/data/*.yaml
        data_dir = self.config_root / "data"
        if data_dir.exists():
            for yaml_file in data_dir.glob("*.yaml"):
                self.data_configs[yaml_file.stem] = self._load_yaml(yaml_file)

        # carica configurazioni modello in core/configs/model/*.yml
        model_dir = self.config_root / "model"
        if model_dir.exists():
            for yml_file in model_dir.glob("*.yaml"):
                self.model_configs[yml_file.stem] = self._load_yaml(yml_file)

        # carica configurazioni backend
        backend_dir = self.config_root / "backend"
        if backend_dir.exists():
            for yml_file in backend_dir.glob("*.yaml"):
                self.backend_configs[yml_file.stem] = self._load_yaml(yml_file)

    @property
    def paths(self) -> Dict[str, Path]:
        """Ritorna i path principali del progetto convertiti in oggetti Path."""
        raw_paths = self.global_cfg.get("paths", {})
        return {k: Path(v) for k, v in raw_paths.items()}

    def get_data_config(self, dataset_name: str = "nci") -> Dict[str, Any]:
        """Estrae la configurazione di un dataset specifico (es. 'nci')."""
        if dataset_name not in self.data_configs:
            raise KeyError(f"Dataset config '{dataset_name}' non trovata tra: {list(self.data_configs.keys())}")
        return self.data_configs[dataset_name]

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Estrae la configurazione di un modello specifico (es. 'egnn3', 'gatv2', 'xgboost')."""
        if model_name not in self.model_configs:
            raise KeyError(f"Model config '{model_name}' non trovata tra: {list(self.model_configs.keys())}")
        return self.model_configs[model_name]

    def get_backend_config(self, backend_name: str) -> Dict[str, Any]:
        """Estrae la configurazione legate ad un backend specifico (es. 'classical', 'neutral_atom_sim', 'neutral_atom_hw')."""
        if backend_name not in self.backend_configs:
            raise KeyError(f"Model config '{backend_name}' non trovata tra: {list(self.backend_configs.keys())}")
        return self.backend_configs[backend_name]    