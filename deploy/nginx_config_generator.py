"""Nginx configuration generator for MiniTrain gray releases.

This module produces an nginx upstream block that gradually shifts traffic from
an old model endpoint to a new model endpoint. Each stage specifies a percentage
of traffic for the new model and a duration. After the final stage the new model
receives 100% of traffic.

The default stage schedule is read from ``config.deploy.gray_stages`` in
``config.yaml``. Example:

    deploy:
      gray_stages:
        - traffic_percent: 10
          duration: "2h"
        - traffic_percent: 50
          duration: "6h"
        - traffic_percent: 100
          duration: "steady"

Gray release strategy:
1. Start by sending a small fraction of traffic to the new model while the
   majority still hits the old (stable) endpoint. Monitor error rates, latency,
   and business metrics.
2. Increase the new model share when the previous stage looks healthy.
3. After the final 100% stage, remove the old endpoint from the upstream.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"
DEPLOY_DIR = Path(__file__).resolve().parent
DEFAULT_GRAY_STAGES: List[Dict[str, Any]] = [
    {"traffic_percent": 10, "duration": "2h"},
    {"traffic_percent": 50, "duration": "6h"},
    {"traffic_percent": 100, "duration": "steady"},
]


def _load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    config_file = path or CONFIG_PATH
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _default_stages(config: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    if config is None:
        try:
            config = _load_config()
        except FileNotFoundError:
            return DEFAULT_GRAY_STAGES
    return config.get("deploy", {}).get("gray_stages", DEFAULT_GRAY_STAGES)


def _weights_for_stage(new_percent: int) -> Tuple[int, int]:
    """Return integer nginx weights for (old, new) given the new-model percent."""
    old_percent = 100 - new_percent
    return old_percent, new_percent


def generate_config(
    old_model_endpoint: str,
    new_model_endpoint: str,
    stage_config: Optional[List[Dict[str, Any]]] = None,
    output_dir: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Generate an nginx upstream snippet and a JSON gray-release plan.

    Parameters
    ----------
    old_model_endpoint:
        Existing stable upstream endpoint, e.g. ``127.0.0.1:8001``.
    new_model_endpoint:
        New model upstream endpoint, e.g. ``127.0.0.1:8002``.
    stage_config:
        Ordered list of stages. Each stage is ``{"traffic_percent": int, "duration": str}``.
        Defaults to ``config.deploy.gray_stages``.
    output_dir:
        Directory where ``gray_config.conf`` and ``gray_plan.json`` are written.
        Defaults to the directory containing this module.

    Returns
    -------
    ``(nginx_config_str, gray_plan_dict)``.
    """
    stages = stage_config or _default_stages()
    out_dir = Path(output_dir) if output_dir else DEPLOY_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    plan_stages: List[Dict[str, Any]] = []
    config_lines: List[str] = [
        "# MiniTrain gray-release nginx upstream configuration",
        "#",
        "# Strategy: gradually shift traffic from the old model to the new model.",
        "# Each stage below lists the percent of traffic routed to the new endpoint",
        "# and the recommended observation duration before moving to the next stage.",
        "#",
    ]

    for idx, stage in enumerate(stages, start=1):
        new_percent = int(stage["traffic_percent"])
        duration = stage["duration"]
        old_weight, new_weight = _weights_for_stage(new_percent)

        plan_stages.append({
            "stage": idx,
            "new_traffic_percent": new_percent,
            "duration": duration,
            "old_weight": old_weight,
            "new_weight": new_weight,
        })

        config_lines.append(f"# Stage {idx}: {new_percent}% new model for {duration}")
        config_lines.append("upstream minitrain_model {")
        if old_weight > 0:
            config_lines.append(f"    server {old_model_endpoint} weight={old_weight};")
        config_lines.append(f"    server {new_model_endpoint} weight={new_weight};")
        config_lines.append("}")
        config_lines.append("")

    gray_plan = {
        "old_model_endpoint": old_model_endpoint,
        "new_model_endpoint": new_model_endpoint,
        "stages": plan_stages,
    }

    nginx_config = "\n".join(config_lines)

    # Persist artifacts.
    nginx_path = out_dir / "gray_config.conf"
    plan_path = out_dir / "gray_plan.json"
    with open(nginx_path, "w", encoding="utf-8") as f:
        f.write(nginx_config)
    with open(plan_path, "w", encoding="utf-8") as f:
        json.dump(gray_plan, f, indent=2)

    return nginx_config, gray_plan


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate MiniTrain gray-release nginx config")
    parser.add_argument("--old-endpoint", required=True, help="Old/stable model endpoint")
    parser.add_argument("--new-endpoint", required=True, help="New model endpoint")
    parser.add_argument("--output-dir", default=None, help="Directory for generated files")
    args = parser.parse_args()

    config_str, plan = generate_config(
        old_model_endpoint=args.old_endpoint,
        new_model_endpoint=args.new_endpoint,
        output_dir=args.output_dir,
    )
    print(config_str)
    print("---")
    print(json.dumps(plan, indent=2))
