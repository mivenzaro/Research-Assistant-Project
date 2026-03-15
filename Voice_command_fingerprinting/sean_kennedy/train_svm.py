#!/usr/bin/env python3
import os
import csv
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn import svm
from sklearn.preprocessing import StandardScaler

def list_csv(dirpath):
    out = []
    for root,_,files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith(".csv"):
                out.append(os.path.join(root, f))
    out.sort()
    return out

def parse_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        reader = csv.reader(fh)
        first = None
        try:
            first = next(reader)
        except StopIteration:
            return []
        hdr = [h.strip().lower() for h in first]
        if "size" in hdr and "direction" in hdr and "time" in hdr:
            size_i = hdr.index("size")
            dir_i = hdr.index("direction")
            time_i = hdr.index("time")
            for r in reader:
                if len(r) <= max(size_i, dir_i, time_i):
                    continue
                try:
                    t = float(r[time_i])
                    s = int(r[size_i])
                    d = int(r[dir_i])
                    rows.append((t, s, d))
                except:
                    continue
        else:
            fh.seek(0)
            for r in reader:
                if len(r) < 4:
                    continue
                try:
                    t = float(r[1])
                    s = int(r[2])
                    d = int(r[3])
                    rows.append((t, s, d))
                except:
                    continue
    return rows

def bursts_from_rows(rows, gap_threshold=0.02):
    if not rows:
        return []
    rows_sorted = sorted(rows, key=lambda x: x[0])
    bursts = []
    current = [rows_sorted[0]]
    for prev,cur in zip(rows_sorted, rows_sorted[1:]):
        if cur[0] - prev[0] > gap_threshold:
            bursts.append(current)
            current = [cur]
        else:
            current.append(cur)
    bursts.append(current)
    burst_sizes = [sum(abs(r[1]) for r in b) for b in bursts]
    burst_counts = len(bursts)
    burst_durations = [b[-1][0] - b[0][0] for b in bursts]
    return burst_sizes, burst_counts, burst_durations

def feature_stats(values):
    if not values:
        return [0.0]*6
    a = np.array(values, dtype=float)
    return [
        float(a.mean()),
        float(np.median(a)),
        float(a.std(ddof=0)),
        float(np.percentile(a, 25)),
        float(np.percentile(a, 75)),
        float(a.max())
    ]

def extract_features(path):
    rows = parse_rows(path)
    if not rows:
        return None
    times = [r[0] for r in rows]
    sizes = [r[1] for r in rows]
    dirs = [r[2] for r in rows]
    incoming = [s for s,d in zip(sizes, dirs) if d < 0]
    outgoing = [s for s,d in zip(sizes, dirs) if d >= 0]
    total_bytes_in = sum(abs(s) for s in incoming)
    total_bytes_out = sum(abs(s) for s in outgoing)
    pkt_count = len(rows)
    in_ratio = (len(incoming) / pkt_count) if pkt_count else 0.0
    size_stats_all = feature_stats(sizes)
    size_stats_in = feature_stats(incoming)
    size_stats_out = feature_stats(outgoing)
    burst_sizes, burst_count, burst_durations = bursts_from_rows(rows)
    burst_stats = feature_stats(burst_sizes)
    avg_burst_duration = float(np.mean(burst_durations)) if burst_durations else 0.0
    inter_arrival = np.diff(sorted(times)) if len(times) > 1 else np.array([0.0])
    ia_stats = [float(inter_arrival.mean()), float(np.median(inter_arrival)), float(inter_arrival.std(ddof=0))]
    feat = [
        total_bytes_in, total_bytes_out, pkt_count, in_ratio, burst_count, avg_burst_duration
    ]
    feat += size_stats_all + size_stats_in + size_stats_out + burst_stats + ia_stats
    return np.array(feat, dtype=float)

def infer_label(path):
    name = Path(path).stem
    if "_burst_" in name:
        name = name.split("_burst_")[0]
    if "_" in name:
        return name.split("_", 1)[0]
    return Path(path).parent.name or name

def build_dataset(dirpath):
    files = list_csv(dirpath)
    X = []
    y = []
    bad = []
    for f in files:
        v = extract_features(f)
        if v is None:
            bad.append(f)
            continue
        X.append(v)
        y.append(infer_label(f))
    if not X:
        raise SystemExit("no usable files")
    return np.vstack(X), np.array(y, dtype=object), bad

def parse_skip_arg(s):
    return set()  # placeholder to keep signature similar (no-op)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("-t","--train", required=True)
    p.add_argument("-o","--out", required=True)
    p.add_argument("--kernel", choices=("linear","rbf","poly","sigmoid"), default="rbf")
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--normalize", action="store_true")
    a = p.parse_args()

    X, y, bad = build_dataset(a.train)
    if bad:
        print("skipped", len(bad), "files")
    scaler = None
    if a.normalize:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    clf = svm.SVC(kernel=a.kernel, C=a.C, probability=True)
    clf.fit(X, y)
    pkg = {"model": clf, "scaler": scaler}
    joblib.dump(pkg, a.out)
    print("model saved:", a.out)

if __name__ == "__main__":
    main()
