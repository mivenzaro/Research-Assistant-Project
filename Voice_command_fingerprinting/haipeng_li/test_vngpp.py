#!/usr/bin/env python3
import os
import sys
import csv
import argparse
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report

def try_float(s):
    try: return float(s)
    except: return None

def try_int(s):
    try: return int(float(s))
    except: return None

def normalize_direction(val):
    if val is None: return None
    if isinstance(val,(int,float)):
        if int(val)==1: return 1
        if int(val)==-1 or int(val)==0: return -1
    s=str(val).strip().lower()
    if s in ('1','1.0','+1','up','uplink','out','outgoing','tx','client','send','sent'): return 1
    if s in ('-1','-1.0','down','dn','in','incoming','rx','server','recv','received','0'): return -1
    n=try_int(s)
    if n is not None:
        if n==1: return 1
        if n==-1 or n==0: return -1
    return None

def read_packets_from_csv(path):
    with open(path, 'r', newline='', encoding='utf-8', errors='ignore') as fh:
        sample = fh.read(8192)
        if not sample:
            raise ValueError("empty file")
        fh.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=[',',';','\t',' '])
        except:
            dialect = csv.get_dialect('excel')
        reader = csv.reader(fh, dialect)
        rows = [r for r in reader if any(cell.strip() for cell in r)]
        if not rows: raise ValueError("no non-empty rows")
        header_like = any(any(ch.isalpha() for ch in cell) for cell in rows[0])
        start = 1 if header_like else 0
        if start >= len(rows): raise ValueError("no data rows")
        N = min(8, len(rows)-start)
        candidates = rows[start:start+N]
        ncols = max(len(r) for r in candidates)
        if ncols < 3: raise ValueError("not enough columns")
        best = None
        best_score = -1
        for t in range(ncols):
            for s in range(ncols):
                if s == t: continue
                for d in range(ncols):
                    if d in (t,s): continue
                    score = 0
                    good = True
                    for r in candidates:
                        row = list(r) + ['']*(ncols - len(r))
                        ft = try_float(row[t])
                        fs = try_int(row[s])
                        fd = normalize_direction(row[d])
                        if ft is not None: score += 2
                        if fs is not None and fs >= 0: score += 2
                        if fd is not None: score += 3
                        if ft is None and fs is None and fd is None:
                            good = False
                            break
                    if not good: continue
                    if score > best_score:
                        best_score = score
                        best = (t,s,d)
        if best is None:
            fallback_orders = [(1,2,3),(0,1,2),(0,2,3),(2,3,1),(1,0,2)]
            chosen = None
            for (t,s,d) in fallback_orders:
                if max(t,s,d) < ncols:
                    chosen = (t,s,d); break
            if chosen is None: raise ValueError("could not detect columns")
            t_idx,s_idx,d_idx = chosen
        else:
            t_idx,s_idx,d_idx = best
        packets = []
        for row in rows[start:]:
            row = list(row) + ['']*(ncols - len(row))
            ts = try_float(row[t_idx])
            size = try_int(row[s_idx])
            direc = normalize_direction(row[d_idx])
            if ts is None or size is None or direc is None: continue
            packets.append((float(ts), int(size), int(direc)))
        if not packets: raise ValueError("no valid packet rows parsed")
        return packets

def round_value(v,bucket):
    if not bucket or bucket<=0: return int(v)
    return int(round(v/bucket)*bucket)

def calculate_bursts_from_packets(packets,round_burst_bucket):
    if not packets: return []
    packets_sorted = sorted(packets,key=lambda x:x[0])
    burst_list=[]
    prev_dir = packets_sorted[0][2]
    tmp_burst = packets_sorted[0][1] * prev_dir
    for _, size, d in packets_sorted[1:]:
        if d == prev_dir:
            tmp_burst += size * d
        else:
            if round_burst_bucket:
                tmp_burst = round_value(tmp_burst, round_burst_bucket)
            burst_list.append(tmp_burst)
            prev_dir = d
            tmp_burst = size * d
    if round_burst_bucket:
        tmp_burst = round_value(tmp_burst, round_burst_bucket)
    burst_list.append(tmp_burst)
    return burst_list

def make_bins(start,end,interval):
    bins=[]
    v=start
    while v<end:
        bins.append((v,v+interval))
        v+=interval
    bins.append((end,float('inf')))
    return bins

def burst_histogram(bursts,start,end,interval):
    bins = make_bins(start,end,interval)
    counts = [0]*len(bins)
    for b in bursts:
        m = abs(int(b))
        for i,(lo,hi) in enumerate(bins):
            if lo <= m < hi:
                counts[i] += 1
                break
    return counts

def compute_vng_features(packets,start,end,interval,round_burst_bucket):
    packets_sorted = sorted(packets,key=lambda x:x[0])
    up_total = sum(size for _,size,d in packets_sorted if d==1)
    down_total = sum(size for _,size,d in packets_sorted if d==-1)
    times = [t for t,_,_ in packets_sorted]
    total_time = max(times)-min(times) if times else 0.0
    bursts = calculate_bursts_from_packets(packets,round_burst_bucket)
    hist = burst_histogram(bursts,start,end,interval)
    return [total_time, up_total, down_total] + hist

def compute_llnb_set(packets,round_packet_bucket):
    vals=set()
    for _,size,d in packets:
        s=round_value(size,round_packet_bucket)
        vals.add(int(s)*int(d))
    return vals

def load_all_traces(indir):
    files=[]
    if os.path.isfile(indir):
        return [indir]
    for root,_,filenames in os.walk(indir):
        for fn in filenames:
            if fn.lower().endswith(('.csv','.txt')):
                files.append(os.path.join(root, fn))
    files.sort()
    return files

def infer_label_from_path(path,method):
    if method=='dirname': return os.path.basename(os.path.dirname(path))
    fname=os.path.splitext(os.path.basename(path))[0]
    if "_burst_" in fname:
        base = fname.split("_burst_")[0]
    else:
        base = fname
    if base.endswith("_5_30s"):
        base = base.rsplit("_5_30s",1)[0]
    return base

def build_test_dataset(paths,label_from,range_start,range_end,range_interval,round_packet,round_burst):
    files_ok=[]
    labels=[]
    X_vng=[]
    X_llnb_sets=[]
    for p in paths:
        try:
            pkts = read_packets_from_csv(p)
        except Exception:
            continue
        vng_feats = compute_vng_features(pkts,range_start,range_end,range_interval,round_burst)
        llnb_set = compute_llnb_set(pkts,round_packet)
        files_ok.append(p)
        labels.append(infer_label_from_path(p,label_from))
        X_vng.append(vng_feats)
        X_llnb_sets.append(llnb_set)
    if not files_ok:
        raise RuntimeError("No valid traces found")
    return files_ok,labels,np.array(X_vng,dtype=float),X_llnb_sets

def main():
    p = argparse.ArgumentParser()
    p.add_argument('-m','--modelFile',required=True)
    p.add_argument('-i','--testPath',required=True)
    p.add_argument('--label-from',choices=('filename','dirname'),default='filename')
    p.add_argument('--range-start',type=int,default=0)
    p.add_argument('--range-end',type=int,default=1500)
    p.add_argument('--range-interval',type=int,default=100)
    p.add_argument('--round-packet',type=int,default=0)
    p.add_argument('--round-burst',type=int,default=0)
    p.add_argument('--print-per-file',action='store_true')
    args = p.parse_args()

    model_obj = joblib.load(args.modelFile)

    model_type=None
    clf=None
    mlb=None
    params={}

    if isinstance(model_obj,tuple) and len(model_obj)>=2:
        model_type=model_obj[0]
        if model_type=='vng':
            clf=model_obj[1]
            if len(model_obj)>2 and isinstance(model_obj[2],dict):
                params=model_obj[2]
        elif model_type=='ll-nb':
            mlb,clf=model_obj[1]
            if len(model_obj)>2 and isinstance(model_obj[2],dict):
                params=model_obj[2]
    else:
        clf=model_obj
        model_type='vng'

    range_start=int(params.get('range_start',args.range_start))
    range_end=int(params.get('range_end',args.range_end))
    range_interval=int(params.get('range_interval',args.range_interval))
    round_packet=params.get('round_packet',args.round_packet)
    round_burst=params.get('round_burst',args.round_burst)

    test_paths=load_all_traces(args.testPath)

    files,labels,X_vng,X_llnb_sets = build_test_dataset(
        test_paths,args.label_from,
        range_start,range_end,range_interval,
        (round_packet or None),(round_burst or None)
    )

    y_true=np.array(labels)

    if model_type=='vng':
        preds=clf.predict(X_vng)
    elif model_type=='ll-nb':
        Xte=mlb.transform(X_llnb_sets)
        preds=clf.predict(Xte)
    else:
        preds=clf.predict(X_vng)

    if args.print_per_file:
        for f,p in zip(files,preds):
            print(f"{os.path.basename(f)} -> {p}")

    acc=accuracy_score(y_true,preds)
    print(f"Predicted {len(preds)} traces. Accuracy: {acc:.4f}")
    print(classification_report(y_true,preds,digits=4,zero_division=0))

if __name__ == '__main__':
    main()
