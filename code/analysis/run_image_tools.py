"""Run configured collector, spatial and density methods on external images."""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import os
from pathlib import Path


TASK_MODULES = {
    "collector": "collector",
    "vertical": "spatial",
    "radial": "spatial",
    "angle": "spatial",
    "flatten": "spatial",
    "density": "density",
}
PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def resolve_input(inputs: dict, key: str | None, data_root: Path) -> str | None:
    if key is None:
        return None
    value = Path(inputs[key]).expanduser()
    if not value.is_absolute():
        value = data_root / value
    if not value.exists():
        raise FileNotFoundError(
            f"'{key}' at {value}; use --input or edit the image configuration"
        )
    return str(value)


def run(
    task: str,
    configuration: dict,
    output_directory: Path,
    data_root: Path | None = None,
) -> dict:
    """Resolve paths and call the source method without changing its numerical settings."""
    from package_paths import INPUT_ROOT, relative_metadata

    input_root = data_root if data_root is not None else INPUT_ROOT
    inputs = configuration["inputs"]
    settings = copy.deepcopy(configuration[task])
    general = {
        "voxel_size_mm": configuration["voxel_size_mm"],
        "pixel_size_mm": settings.get("pixel_size_mm", configuration["voxel_size_mm"]),
        "seed": configuration["seed"],
        "bootstrap_seed": configuration["seed"],
        "bootstrap_samples": configuration["bootstrap_samples"],
        "bootstrap_block": configuration["block_length"],
    }
    if task == "collector":
        settings["slices_path"] = resolve_input(inputs, settings["input"], input_root)
        settings["reference_path"] = resolve_input(
            inputs, settings["reference_input"], input_root
        )
    elif task == "vertical":
        settings["defect_dir"] = resolve_input(
            inputs, settings["defect_input"], input_root
        )
        settings["collector_path"] = resolve_input(
            inputs, settings["collector_input"], input_root
        )
        settings["exposure_path"] = resolve_input(
            inputs, settings["exposure_input"], input_root
        )
    elif task in ("radial", "angle"):
        settings["defect_dir"] = resolve_input(inputs, settings["input"], input_root)
        settings["exposure_path"] = resolve_input(
            inputs, settings["exposure_input"], input_root
        )
    elif task == "flatten":
        settings["flatten_path"] = resolve_input(inputs, settings["input"], input_root)
        settings["exposure_path"] = resolve_input(
            inputs, settings["exposure_input"], input_root
        )
    elif task == "density":
        settings["defect_path"] = resolve_input(inputs, settings["input"], input_root)
        settings["exposure_path"] = resolve_input(
            inputs, settings["exposure_input"], input_root
        )
        settings["sigmas_mm"] = settings["sigma_mm"]
    from athena_analysis.common import json_safe

    output_directory.mkdir(parents=True, exist_ok=True)
    module = importlib.import_module(f"athena_analysis.{TASK_MODULES[task]}")
    runner = getattr(module, f"run_{task}_analysis")
    result = runner(settings, general, output_directory)
    with (output_directory / "run_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(
            relative_metadata(json_safe(result)), stream, indent=2, allow_nan=False
        )
        stream.write("\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=tuple(TASK_MODULES))
    parser.add_argument(
        "--config",
        type=Path,
        default=PACKAGE_ROOT / "configs" / "image_analysis.json",
        help="Method settings; relative input paths are resolved beneath --data-root.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("ATHENA_DATA_ROOT", PACKAGE_ROOT / "data")),
        help="External input parent containing analysis/ and mechanics/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("ATHENA_OUTPUT_ROOT", PACKAGE_ROOT / "outputs")),
        help="Output parent; results are written beneath analysis/images/<task>/.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        help="Override the primary source image or segmented TIFF directory.",
    )
    parser.add_argument(
        "--collector",
        type=Path,
        help="Override the collector reference used by the vertical analysis.",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Override the full task output directory."
    )
    arguments = parser.parse_args()
    data_root = arguments.data_root.expanduser().resolve()
    os.environ["ATHENA_DATA_ROOT"] = str(data_root)
    os.environ["ATHENA_OUTPUT_ROOT"] = str(arguments.output_root.expanduser().resolve())

    from package_paths import OUTPUT_ROOT, missing_input_message

    if arguments.collector is not None and arguments.task != "vertical":
        parser.error("--collector applies to the vertical task.")
    try:
        configuration = json.loads(arguments.config.read_text(encoding="utf-8"))
        settings = configuration[arguments.task]
        primary_key = (
            settings["defect_input"]
            if arguments.task == "vertical"
            else settings["input"]
        )
        if arguments.input is not None:
            configuration["inputs"][primary_key] = str(
                arguments.input.expanduser().resolve()
            )
        if arguments.collector is not None:
            configuration["inputs"][settings["collector_input"]] = str(
                arguments.collector.expanduser().resolve()
            )
        output_directory = (
            arguments.output_dir or OUTPUT_ROOT / "images" / arguments.task
        )
        run(arguments.task, configuration, output_directory, data_root)
    except FileNotFoundError as error:
        parser.exit(2, f"error: {missing_input_message(error)}\n")
    print("Image analysis:", output_directory)


if __name__ == "__main__":
    main()
