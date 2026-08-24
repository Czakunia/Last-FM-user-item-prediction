"""Ładowanie konfiguracji YAML i plików danych KGAT."""

from __future__ import annotations

import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_config(config_path: str | Path = "configs/config.yaml") -> dict[str, Any]:
    return load_yaml(config_path)


def load_schema(schema_path: str | Path | None = None) -> dict[str, Any]:
    if schema_path is None:
        cfg = load_config()
        schema_path = cfg["paths"]["schema"]
    return load_yaml(schema_path)


def dataset_dir(cfg: dict[str, Any], dataset: str | None = None) -> Path:
    name = dataset or cfg["active_dataset"]
    rel = cfg["datasets"][name]["path"]
    return (ROOT / rel).resolve()


def _ensure_kg_txt(data_dir: Path, cfg_ds: dict[str, Any]) -> Path:
    kg_path = data_dir / cfg_ds["kg_file"]
    if kg_path.exists():
        return kg_path
    zip_path = data_dir / f"{cfg_ds['kg_file']}.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(data_dir)
        if kg_path.exists():
            return kg_path
    raise FileNotFoundError(f"Brak pliku KG: {kg_path} (ani {zip_path})")


def load_id_map(path: Path) -> pd.DataFrame:
    """Robust ID map loader.

    Most rows are whitespace-separated, but some Freebase org_ids contain
    spaces/quotes (seen in amazon-book entity_list). We keep the last token
    as remap_id and join the rest as org_id; optional third column = freebase_id.
    """
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        header = f.readline().strip().split()
        has_freebase = len(header) >= 3 and header[-1] == "freebase_id"
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            try:
                remap_id = int(parts[-1] if not has_freebase else parts[-2])
            except ValueError:
                continue
            if has_freebase:
                if len(parts) < 3:
                    continue
                freebase_id = parts[-1]
                org_id = " ".join(parts[:-2])
                rows.append(
                    {"org_id": org_id, "remap_id": remap_id, "freebase_id": freebase_id}
                )
            else:
                org_id = " ".join(parts[:-1])
                rows.append({"org_id": org_id, "remap_id": remap_id})
    return pd.DataFrame(rows)


def load_interactions(path: Path) -> pd.DataFrame:
    """Zwraca DataFrame: user_id, item_id (jeden wiersz na interakcję)."""
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            user = int(parts[0])
            for item in parts[1:]:
                rows.append((user, int(item)))
    return pd.DataFrame(rows, columns=["user_id", "item_id"])


def load_kg(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=["head", "relation", "tail"],
        dtype={"head": "int32", "relation": "int32", "tail": "int32"},
        engine="c",
    )


def load_dataset(cfg: dict[str, Any] | None = None, dataset: str | None = None) -> dict[str, Any]:
    """Ładuje komplet plików aktywnego (lub wskazanego) datasetu."""
    cfg = cfg or load_config()
    name = dataset or cfg["active_dataset"]
    data_dir = dataset_dir(cfg, name)
    cfg_ds = cfg["datasets"][name]
    kg_path = _ensure_kg_txt(data_dir, cfg_ds)

    users = load_id_map(data_dir / "user_list.txt")
    items = load_id_map(data_dir / "item_list.txt")
    entities = load_id_map(data_dir / "entity_list.txt")
    relations = load_id_map(data_dir / "relation_list.txt")
    train = load_interactions(data_dir / "train.txt")
    test = load_interactions(data_dir / "test.txt")
    kg = load_kg(kg_path)

    return {
        "name": name,
        "path": data_dir,
        "users": users,
        "items": items,
        "entities": entities,
        "relations": relations,
        "train": train,
        "test": test,
        "kg": kg,
    }


def interaction_degree_maps(
    interactions: pd.DataFrame,
) -> tuple[dict[int, int], dict[int, int]]:
    user_deg = interactions.groupby("user_id").size().to_dict()
    item_deg = interactions.groupby("item_id").size().to_dict()
    return user_deg, item_deg


def kg_degree_maps(kg: pd.DataFrame) -> dict[str, dict[int, int]]:
    out_deg: dict[int, int] = defaultdict(int)
    in_deg: dict[int, int] = defaultdict(int)
    for h, t in zip(kg["head"].to_numpy(), kg["tail"].to_numpy()):
        out_deg[int(h)] += 1
        in_deg[int(t)] += 1
    all_nodes = set(out_deg) | set(in_deg)
    total = {n: out_deg.get(n, 0) + in_deg.get(n, 0) for n in all_nodes}
    return {"out": dict(out_deg), "in": dict(in_deg), "total": total}
