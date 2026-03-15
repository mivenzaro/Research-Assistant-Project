import os
import csv
import math
import argparse
from collections import defaultdict

BIN_SIZE = 10
SKIP_SIZES = set()


def get_label(filename):
    name = os.path.splitext(os.path.basename(filename))[0]
    if "_burst_" in name:
        return name.split("_burst_")[0]
    return name


def csv_to_set(csv_path):
    values = set()
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            size = int(row["size"])
            direction = int(row["direction"])
            if size in SKIP_SIZES:
                continue
            rounded_size = round(size / BIN_SIZE) * BIN_SIZE
            values.add(rounded_size * direction)
    return values


def group_files_by_label(input_dir):
    grouped = defaultdict(list)
    for file_name in os.listdir(input_dir):
        if file_name.endswith(".csv"):
            label = get_label(file_name)
            grouped[label].append(os.path.join(input_dir, file_name))
    return grouped


def build_class_signature(file_list):
    sets = [csv_to_set(f) for f in file_list]
    threshold = math.ceil(len(sets) / 2)

    all_values = set()
    for s in sets:
        all_values |= s

    signature = set()
    for value in all_values:
        count = sum(1 for s in sets if value in s)
        if count >= threshold:
            signature.add(value)

    return signature


def save_signatures(output_dir, signatures):
    os.makedirs(output_dir, exist_ok=True)
    for label, signature in signatures.items():
        out_path = os.path.join(output_dir, f"{label}.txt")
        with open(out_path, "w", encoding="utf-8") as f:
            for value in sorted(signature):
                f.write(f"{value}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--train_dir", required=True)
    parser.add_argument("-o", "--output_dir", required=True)
    args = parser.parse_args()

    grouped = group_files_by_label(args.train_dir)
    signatures = {}

    for label in sorted(grouped.keys()):
        files = grouped[label]
        signatures[label] = build_class_signature(files)
        print(f"{label}: {len(files)} files -> {len(signatures[label])} signature values")

    save_signatures(args.output_dir, signatures)
    print("Training complete.")


if __name__ == "__main__":
    main()



