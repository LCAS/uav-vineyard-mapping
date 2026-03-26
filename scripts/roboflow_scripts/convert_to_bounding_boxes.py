#!/usr/bin/env python3

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple


@dataclass
class ConversionStats:
	total_files: int = 0
	total_lines: int = 0
	kept_lines: int = 0
	removed_class_1: int = 0
	removed_class_2: int = 0
	converted_polygons: int = 0
	bbox_already: int = 0
	skipped_invalid: int = 0


def parse_line(line: str) -> Tuple[int, List[float]] | None:
	stripped = line.strip()
	if not stripped:
		return None
	parts = stripped.split()
	if len(parts) < 5:
		return None
	try:
		class_id = int(float(parts[0]))
		coords = [float(v) for v in parts[1:]]
	except ValueError:
		return None
	return class_id, coords


def polygon_to_bbox(coords: List[float]) -> Tuple[float, float, float, float] | None:
	if len(coords) % 2 != 0:
		return None
	xs = coords[0::2]
	ys = coords[1::2]
	if not xs or not ys:
		return None
	x_min = min(xs)
	x_max = max(xs)
	y_min = min(ys)
	y_max = max(ys)
	xc = (x_min + x_max) / 2.0
	yc = (y_min + y_max) / 2.0
	w = x_max - x_min
	h = y_max - y_min
	return xc, yc, w, h


def format_bbox(class_id: int, bbox: Tuple[float, float, float, float]) -> str:
	xc, yc, w, h = bbox
	return f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"


def process_label_file(label_path: Path, output_label_path: Path, stats: ConversionStats) -> None:
	stats.total_files += 1
	output_lines: List[str] = []

	for line in label_path.read_text().splitlines():
		parsed = parse_line(line)
		if parsed is None:
			stats.skipped_invalid += 1
			continue

		class_id, coords = parsed
		stats.total_lines += 1

		if class_id == 1:
			stats.removed_class_1 += 1
			continue
		if class_id == 2:
			stats.removed_class_2 += 1
			continue

		if class_id == 0:
			if len(coords) == 4:
				stats.bbox_already += 1
				output_lines.append(format_bbox(class_id, tuple(coords)))
			else:
				bbox = polygon_to_bbox(coords)
				if bbox is None:
					stats.skipped_invalid += 1
					continue
				stats.converted_polygons += 1
				output_lines.append(format_bbox(class_id, bbox))
			stats.kept_lines += 1
			continue

		stats.kept_lines += 1
		output_lines.append(line.strip())

	output_label_path.parent.mkdir(parents=True, exist_ok=True)
	output_label_path.write_text("\n".join(output_lines) + ("\n" if output_lines else ""))


def iter_label_files(dataset_dir: Path) -> Iterable[Tuple[Path, Path]]:
	for split in ("train", "valid", "test"):
		labels_dir = dataset_dir / split / "labels"
		if not labels_dir.exists():
			continue
		for label_path in sorted(labels_dir.glob("*.txt")):
			rel_path = label_path.relative_to(dataset_dir)
			yield label_path, rel_path


def run(dataset_dir: Path, output_dir: Path) -> ConversionStats:
	stats = ConversionStats()
	for label_path, rel_path in iter_label_files(dataset_dir):
		output_label_path = output_dir / rel_path
		process_label_file(label_path, output_label_path, stats)
	return stats


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(
		description=(
			"Remove classes 1 (trunks) and 2 (vine_row) and convert class 0 polygons "
			"to YOLO bounding boxes."
		)
	)
	parser.add_argument(
		"--dataset-dir",
		type=Path,
		default=Path(
			"../../data/datasets/vineyard_segmentation_paper-65"
		),
		help="Path to the dataset root containing train/valid/test folders.",
	)
	parser.add_argument(
		"--output-dir",
		type=Path,
		default=Path(
			"../../data/datasets/vineyard_segmentation_paper-65_converted"
		),
		help="Output dataset root (labels will be written here with same structure).",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()
	dataset_dir = args.dataset_dir.expanduser().resolve()
	output_dir = args.output_dir.expanduser().resolve()
	stats = run(dataset_dir, output_dir)

	print("Done.")
	print(f"Dataset: {dataset_dir}")
	print(f"Output: {output_dir}")
	print(f"Files processed: {stats.total_files}")
	print(f"Total labels read: {stats.total_lines}")
	print(f"Kept labels: {stats.kept_lines}")
	print(f"Removed class 1: {stats.removed_class_1}")
	print(f"Removed class 2: {stats.removed_class_2}")
	print(f"Converted polygons: {stats.converted_polygons}")
	print(f"Already bbox: {stats.bbox_already}")
	print(f"Skipped invalid: {stats.skipped_invalid}")


if __name__ == "__main__":
	main()
