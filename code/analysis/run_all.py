"""Run the study's saved-table analyses using external input data."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task",
        choices=("all", "fixed_center", "nodes", "axial_ends", "validation"),
        default="all",
    )
    package_root = Path(__file__).resolve().parents[2]
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("ATHENA_DATA_ROOT", package_root / "data")),
        help="External input parent containing analysis/ and mechanics/.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(os.environ.get("ATHENA_OUTPUT_ROOT", package_root / "outputs")),
        help="Output parent; analysis results are written beneath analysis/.",
    )
    arguments = parser.parse_args()
    os.environ["ATHENA_DATA_ROOT"] = str(arguments.data_root.expanduser().resolve())
    os.environ["ATHENA_OUTPUT_ROOT"] = str(arguments.output_root.expanduser().resolve())

    from package_paths import missing_input_message

    try:
        if arguments.task in ("all", "fixed_center"):
            import fixed_center_statistics
            import rebuild_saved_fit

            rebuild_saved_fit.run()
            fixed_center_statistics.run()
        if arguments.task in ("all", "nodes"):
            import node_alignment

            node_alignment.run()
        if arguments.task in ("all", "axial_ends"):
            import axial_ends

            axial_ends.main()
        if arguments.task in ("all", "validation"):
            import validation_metrics

            validation_metrics.main()
    except FileNotFoundError as error:
        parser.exit(2, f"error: {missing_input_message(error)}\n")


if __name__ == "__main__":
    main()
