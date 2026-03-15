import os
import csv
import argparse

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


def load_signatures(signature_dir):
    signatures = {}
    for file_name in os.listdir(signature_dir):
        if file_name.endswith(".txt"):
            label = os.path.splitext(file_name)[0]
            path = os.path.join(signature_dir, file_name)
            with open(path, "r", encoding="utf-8") as f:
                signatures[label] = set(int(line.strip()) for line in f if line.strip())
    return signatures


def jaccard_similarity(a, b):
    if not a and not b:
        return 1.0
    union = a | b
    inter = a & b
    return len(inter) / len(union) if union else 0.0


def predict(csv_path, signatures):
    test_set = csv_to_set(csv_path)
    best_label = None
    best_score = -1.0

    for label, signature in signatures.items():
        score = jaccard_similarity(test_set, signature)
        if score > best_score:
            best_score = score
            best_label = label

    return best_label, best_score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--signature_dir", required=True)
    parser.add_argument("-i", "--input_dir", required=True)
    args = parser.parse_args()

    signatures = load_signatures(args.signature_dir)

    total = 0
    correct = 0

    for file_name in sorted(os.listdir(args.input_dir)):
        if not file_name.endswith(".csv"):
            continue

        csv_path = os.path.join(args.input_dir, file_name)
        true_label = get_label(file_name)
        pred_label, score = predict(csv_path, signatures)

        total += 1
        if pred_label == true_label:
            correct += 1

        print(f"{file_name} -> true: {true_label}, pred: {pred_label}, score: {score:.4f}")

    accuracy = correct / total if total else 0.0
    print(f"\nAccuracy: {correct}/{total} = {accuracy:.4f}")


if __name__ == "__main__":
    main()
