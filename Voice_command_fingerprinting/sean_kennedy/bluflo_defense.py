#!/usr/bin/env python3
import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

def read_trace(csv_path: Path) -> List[Tuple[int, float, int, int]]:
    df = pd.read_csv(csv_path)
    col_map = {c.lower(): c for c in df.columns}
    if 'time' not in col_map or 'size' not in col_map or 'direction' not in col_map:
        raise ValueError("CSV must contain 'time', 'size', and 'direction' columns")
    times = df[col_map['time']].astype(float).tolist()
    sizes = df[col_map['size']].astype(int).tolist()
    dirs = df[col_map['direction']].astype(float).tolist()
    trace = []
    for i, (tt, sz, dr) in enumerate(zip(times, sizes, dirs)):
        sign = 1 if float(dr) >= 0 else -1
        trace.append((i, float(tt), int(sz), sign))
    return trace

def chunk_packet(size: int, d: int) -> Tuple[List[int], int]:
    if size <= d:
        return [size], d - size
    full = size // d
    rem = size % d
    chunks = [d] * full
    if rem:
        chunks.append(rem)
    overhead = sum(d - c for c in chunks)
    return chunks, overhead

def apply_buflo_ordered(trace: List[Tuple[int, float, int, int]], d: int, f: float, t: float, seed: int = None):
    if seed is not None:
        random.seed(seed)
    total_original_bytes = sum(p[2] for p in trace)
    start_t = trace[0][1]
    original_end_time = trace[-1][1]
    min_total_packets = max(0, int(math.ceil(t * f)))
    obf = []
    next_index = 0
    for orig_idx, orig_time, orig_size, orig_dir in trace:
        chunks, overhead_for_this_packet = chunk_packet(orig_size, d)
        for i_chunk, chunk_size in enumerate(chunks):
            if chunk_size == orig_size and len(chunks) == 1:
                if chunk_size < d:
                    overhead = d - chunk_size
                    status = "padded"
                    out_size = d
                else:
                    overhead = 0
                    status = "unchanged"
                    out_size = chunk_size
            else:
                if chunk_size == d:
                    overhead = 0
                    status = "chopped" if orig_size > d else "new"
                    out_size = d
                else:
                    overhead = d - chunk_size
                    status = "chopped_padded"
                    out_size = d
            pkt_time = start_t + next_index * (1.0 / f)
            obf.append([next_index, round(pkt_time, 6), out_size, orig_dir, overhead, status])
            next_index += 1
    while next_index < min_total_packets:
        pkt_time = start_t + next_index * (1.0 / f)
        direction = 1 if random.random() >= 0.5 else -1
        obf.append([next_index, round(pkt_time, 6), d, direction, d, "dummy"])
        next_index += 1
    total_out_bytes = sum(row[2] for row in obf)
    time_delay_seconds = obf[-1][1] - original_end_time if len(obf) > 0 else 0.0
    return obf, total_original_bytes, total_out_bytes, time_delay_seconds

def write_buflo_csv(out_path: Path, obf_packets: List[List]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(obf_packets, columns=["index", "time", "size", "direction", "overhead", "status"])
    df.to_csv(out_path, index=False)

def append_info(info_file: Path, trace_name: str, original_bytes: int, out_bytes: int, overhead: int, time_delay: float):
    header = ["trace_name", "total_original_bytes", "total_out_bytes", "total_overhead", "time_delay_seconds"]
    exists = info_file.exists()
    info_file.parent.mkdir(parents=True, exist_ok=True)
    with open(info_file, "a", newline="") as fh:
        writer = csv.writer(fh)
        if not exists:
            writer.writerow(header)
        writer.writerow([trace_name, original_bytes, out_bytes, out_bytes - original_bytes, round(time_delay, 6)])

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("csv_path", type=str)
    p.add_argument("--d", type=int, required=True)
    p.add_argument("--f", type=float, required=True)
    p.add_argument("--t", type=float, required=True)
    p.add_argument("--out-dir", type=str, default="./buflo_out")
    p.add_argument("--info-file", type=str, default="overhead_info.csv")
    p.add_argument("--seed", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    path = Path(args.csv_path)
    d = int(args.d)
    f = float(args.f)
    t = float(args.t)
    seed = args.seed
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if path.is_dir():
        files = sorted(path.glob("*.csv"))
    elif path.is_file():
        files = [path]
    else:
        raise FileNotFoundError(f"No file or directory at {path}")

    for csv_file in files:
        print(f"[+] Reading: {csv_file}")
        trace = read_trace(csv_file)
        trace_name = csv_file.stem
        obf_packets, orig_bytes, out_bytes, time_delay = apply_buflo_ordered(trace, d=d, f=f, t=t, seed=seed)
        out_path = out_dir / f"{trace_name}_buflo.csv"
        write_buflo_csv(out_path, obf_packets)
        append_info(Path(args.info_file), trace_name, orig_bytes, out_bytes, out_bytes - orig_bytes, time_delay)
        print(f"[+] Wrote: {out_path}")

if __name__ == "__main__":
    main()

