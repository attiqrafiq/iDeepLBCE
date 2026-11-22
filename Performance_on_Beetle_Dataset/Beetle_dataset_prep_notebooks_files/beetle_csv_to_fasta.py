"""Convert BeeTLe split CSVs to binary-labeled FASTA files."""

import argparse
import csv
import re
from pathlib import Path


def convert_splits(csv_dir, output_dir):
    """Preserve split membership and order; recode labels 1–4 as positive."""
    csv_dir = Path(csv_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "valid", "test"):
        source = csv_dir / f"{split}.csv"
        destination = output_dir / f"{split}.fasta"
        temporary = destination.with_suffix(".fasta.tmp")
        counts = [0, 0]
        seen = set()

        try:
            with source.open(encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                required = {"Epitope ID", "Description", "Label"}
                if not required.issubset(reader.fieldnames or []):
                    raise ValueError(f"{source} must include {sorted(required)}")

                with temporary.open("w", encoding="ascii", newline="\n") as fasta:
                    for row in reader:
                        epitope_id = (row["Epitope ID"] or "").strip()
                        peptide = (row["Description"] or "").strip()
                        label = (row["Label"] or "").strip()
                        context = f"{source}:{reader.line_num}"

                        if not re.fullmatch(r"[0-9]+", epitope_id) or epitope_id in seen:
                            raise ValueError(f"{context}: invalid or repeated epitope ID {epitope_id!r}")
                        if not re.fullmatch(r"[A-Z]+", peptide):
                            raise ValueError(f"{context}: expected an uppercase peptide sequence")
                        if label not in {"0", "1", "2", "3", "4"}:
                            raise ValueError(f"{context}: unexpected BeeTLe label {label!r}")

                        binary_label = int(label != "0")
                        fasta.write(f">{epitope_id}_{binary_label}\n{peptide}\n")
                        counts[binary_label] += 1
                        seen.add(epitope_id)

            temporary.replace(destination)
        finally:
            temporary.unlink(missing_ok=True)

        print(f"{destination}: {counts[0]:,} non-epitopes, {counts[1]:,} epitopes")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_dir", help="Directory containing train.csv, valid.csv and test.csv")
    parser.add_argument("--output-dir", help="FASTA directory (default: a fasta subdirectory)")
    args = parser.parse_args()
    convert_splits(args.csv_dir, args.output_dir or Path(args.csv_dir) / "fasta")
