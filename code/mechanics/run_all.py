"""Run final mechanics numerical post-processing without starting COMSOL."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
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
        help="Output parent; mechanics results are written beneath mechanics/.",
    )
    arguments = parser.parse_args()
    os.environ["ATHENA_DATA_ROOT"] = str(arguments.data_root.expanduser().resolve())
    os.environ["ATHENA_OUTPUT_ROOT"] = str(arguments.output_root.expanduser().resolve())

    from package_paths import missing_input_message

    try:
        from analyze_bending import main as analyze_bending
        from analyze_cumulative_transfer import main as analyze_cumulative_transfer
        from analyze_end_bending import main as analyze_end_bending
        from analyze_surface_strain import main as analyze_surface_strain

        analyze_bending()
        analyze_end_bending()
        analyze_cumulative_transfer()
        analyze_surface_strain()
    except FileNotFoundError as error:
        parser.exit(2, f"error: {missing_input_message(error)}\n")


if __name__ == "__main__":
    main()
