"""
Afilter_dash.py

Cleans up non-dashcam data (specifically China_Drone and China_MotorBike files)
from the entire RDD2022 dataset in a single run across all splits (train, val, test)
and subdirectories (images, labels).

Usage:
    # Run interactive cleanup on default './rdd2022' (scans, shows breakdown, prompts [y/N]):
    python Afilter_dash.py

    # Non-interactive / force delete (skip confirmation prompt):
    python Afilter_dash.py --yes
    # or
    python Afilter_dash.py -y

    # Dry run (scan and show detailed breakdown only, no prompt, no files deleted):
    python Afilter_dash.py --dry-run

    # Specify a custom directory:
    python Afilter_dash.py --dir ./rdd2022
"""

import argparse
from collections import defaultdict
from pathlib import Path
import sys

EXCLUDE_PREFIXES = ("China_Drone", "China_MotorBike")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Clean up China_Drone and China_MotorBike files from RDD2022 dataset."
    )
    parser.add_argument(
        "--dir",
        default="./rdd2022",
        help="Root dataset directory to clean (default: './rdd2022')",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip interactive confirmation prompt and delete files immediately",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run: scan and report without deleting anything",
    )
    # Backward compatibility alias for the old script
    parser.add_argument(
        "--confirm",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def classify_path(p: Path, root: Path):
    """Categorize file by split, folder type, and matched prefix."""
    try:
        rel_parts = [part.lower() for part in p.relative_to(root).parts]
    except ValueError:
        rel_parts = [part.lower() for part in p.parts]

    # Detect split (train / val / test)
    split = "other"
    for s in ("train", "val", "test"):
        if s in rel_parts:
            split = s
            break

    # Detect category (images / labels / other)
    category = "other"
    if "images" in rel_parts:
        category = "images"
    elif "labels" in rel_parts:
        category = "labels"
    else:
        category = p.suffix.lstrip(".").lower() or "other"

    # Detect matched prefix
    prefix = "Other"
    for pref in EXCLUDE_PREFIXES:
        if p.name.startswith(pref):
            prefix = pref
            break

    return split, category, prefix


def main():
    args = parse_args()
    auto_confirm = args.yes or args.confirm

    root = Path(args.dir).expanduser().resolve()
    if not root.exists():
        sys.exit(f"Error: Target directory does not exist: {root}\nPlease verify the path or specify via --dir.")

    print("=" * 70)
    print(" RDD2022 Dataset Cleanup (China_Drone & China_MotorBike Filter)")
    print("=" * 70)
    print(f"Target Directory: {root}")
    print("Scanning directory tree, please wait...\n")

    total_scanned = 0
    to_delete = []

    # Recursively scan for all matching files
    for p in root.rglob("*"):
        if p.is_file():
            total_scanned += 1
            if p.name.startswith(EXCLUDE_PREFIXES):
                to_delete.append(p)

    print(f"Total files scanned: {total_scanned:,}")
    print(f"Files matched for deletion: {len(to_delete):,}")

    if not to_delete:
        print("\nNo files matching China_Drone or China_MotorBike were found.")
        print("Dataset is already clean.")
        return

    # Aggregate breakdown
    # breakdown[split][category][prefix] = count
    breakdown = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    for p in to_delete:
        split, category, prefix = classify_path(p, root)
        breakdown[split][category][prefix] += 1

    # Print breakdown table
    print("\n" + "-" * 70)
    print(f"{'Split':<10} {'Category':<12} {'China_Drone':<15} {'China_MotorBike':<18} {'Total':<10}")
    print("-" * 70)

    grand_drone = 0
    grand_moto = 0
    grand_total = 0

    known_splits = ["train", "val", "test"] + [s for s in sorted(breakdown.keys()) if s not in ("train", "val", "test")]

    for split in known_splits:
        if split not in breakdown:
            continue
        categories = sorted(breakdown[split].keys())
        for cat in categories:
            drone_count = breakdown[split][cat].get("China_Drone", 0)
            moto_count = breakdown[split][cat].get("China_MotorBike", 0)
            row_total = drone_count + moto_count

            grand_drone += drone_count
            grand_moto += moto_count
            grand_total += row_total

            print(f"{split:<10} {cat:<12} {drone_count:<15,d} {moto_count:<18,d} {row_total:<10,d}")

    print("-" * 70)
    print(f"{'Total':<23} {grand_drone:<15,d} {grand_moto:<18,d} {grand_total:<10,d}")
    print("-" * 70)

    # Show samples
    print("\nSample files marked for deletion:")
    for p in to_delete[:8]:
        try:
            rel = p.relative_to(root)
        except ValueError:
            rel = p
        print(f"  - {rel}")
    if len(to_delete) > 8:
        print(f"  ... and {len(to_delete) - 8:,} more files")

    # Handle dry run
    if args.dry_run:
        print("\n[Dry Run] No files were deleted. Run without --dry-run to delete.")
        return

    # Handle confirmation prompt
    if not auto_confirm:
        print("\n" + "=" * 70)
        try:
            choice = input(f"Are you sure you want to permanently delete these {len(to_delete):,} files? [y/N]: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print("\nAborted by user.")
            return

        if choice not in ("y", "yes"):
            print("Cleanup cancelled. No files were deleted.")
            return

    # Perform deletion
    print(f"\nDeleting {len(to_delete):,} files...")
    deleted_count = 0
    failed_count = 0
    errors = []

    for i, p in enumerate(to_delete, 1):
        try:
            p.unlink()
            deleted_count += 1
        except Exception as e:
            failed_count += 1
            if len(errors) < 10:
                errors.append(f"{p}: {e}")

        # Update progress every 200 files or on the final file
        if i % 200 == 0 or i == len(to_delete):
            pct = (i / len(to_delete)) * 100
            print(f"\rProgress: {i:,}/{len(to_delete):,} files processed ({pct:.1f}%)", end="", flush=True)

    print("\n\n" + "=" * 70)
    print(" Cleanup Complete")
    print("=" * 70)
    print(f"Successfully deleted: {deleted_count:,} files")
    if failed_count > 0:
        print(f"Failed to delete:     {failed_count:,} files")
        print("\nFirst few deletion errors:")
        for err in errors:
            print(f"  - {err}")
    else:
        print("All target files were successfully removed!")
    print("=" * 70)


if __name__ == "__main__":
    main()