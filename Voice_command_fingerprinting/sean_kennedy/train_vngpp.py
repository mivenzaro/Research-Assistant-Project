#!/usr/bin/env python3
import os
import csv
import argparse
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MultiLabelBinarizer

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

def build_datasets(input_dir,label_from,range_start,range_end,range_interval,round_packet,round_burst):
    files = load_all_traces(input_dir)
    X_vng=[]
    X_llnb_sets=[]
    files_ok=[]
    labels=[]
    for fpath in files:
        try:
            pkts = read_packets_from_csv(fpath)
            if not pkts: continue
        except Exception:
            continue
        vng_feats = compute_vng_features(pkts,range_start,range_end,range_interval,round_burst)
        llnb_set = compute_llnb_set(pkts,round_packet)
        X_vng.append(vng_feats)
        X_llnb_sets.append(llnb_set)
        files_ok.append(fpath)
        labels.append(infer_label_from_path(fpath,label_from))
    if not files_ok: raise RuntimeError("No valid traces")
    return files_ok, labels, np.array(X_vng,dtype=float), X_llnb_sets

def evaluate_vng_gaussiannb(X,y,cv_folds):
    skf = StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=42)
    accs=[]
    for tr_idx,te_idx in skf.split(X,y):
        clf = GaussianNB()
        clf.fit(X[tr_idx], y[tr_idx])
        preds = clf.predict(X[te_idx])
        accs.append(accuracy_score(y[te_idx], preds))
    return np.mean(accs), np.std(accs)

def evaluate_llnb_sets(X_sets,y,cv_folds):
    skf = StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=42)
    accs=[]
    for tr_idx,te_idx in skf.split(X_sets,y):
        train_sets = [X_sets[i] for i in tr_idx]
        test_sets = [X_sets[i] for i in te_idx]
        mlb = MultiLabelBinarizer()
        Xtr = mlb.fit_transform(train_sets)
        Xte = mlb.transform(test_sets)
        clf = BernoulliNB()
        clf.fit(Xtr, [y[i] for i in tr_idx])
        preds = clf.predict(Xte)
        accs.append(accuracy_score([y[i] for i in te_idx], preds))
    return np.mean(accs), np.std(accs)

def evaluate_jaccard(X_sets,y,cv_folds):
    skf = StratifiedKFold(n_splits=cv_folds,shuffle=True,random_state=42)
    accs=[]
    for tr_idx,te_idx in skf.split(X_sets,y):
        train_sets = [X_sets[i] for i in tr_idx]
        train_labels = [y[i] for i in tr_idx]
        correct=0
        for i in te_idx:
            s = X_sets[i]
            best_j = -1.0
            best_label = None
            for ts,lab in zip(train_sets,train_labels):
                inter = len(s & ts)
                uni = len(s | ts)
                j = (inter/uni) if uni>0 else 0.0
                if j > best_j:
                    best_j = j
                    best_label = lab
            if best_label == y[i]: correct += 1
        accs.append(correct / len(te_idx))
    return np.mean(accs), np.std(accs)

def save_model(obj,path):
    joblib.dump(obj,path)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir', required=True)
    p.add_argument('--label-from', choices=('filename','dirname'), default='filename')
    p.add_argument('--attack', nargs='+', choices=('vng','ll-nb','jaccard','all'), default=['vng'])
    p.add_argument('--range-start', type=int, default=0)
    p.add_argument('--range-end', type=int, default=1500)
    p.add_argument('--range-interval', type=int, default=100)
    p.add_argument('--round-packet', type=int, default=0)
    p.add_argument('--round-burst', type=int, default=0)
    p.add_argument('--cv', type=int, default=5)
    p.add_argument('--out-model')
    args = p.parse_args()

    do_all = 'all' in args.attack
    attacks = ['vng','ll-nb','jaccard'] if do_all else args.attack

    files, labels, X_vng, X_llnb_sets = build_datasets(
        args.input_dir, args.label_from,
        args.range_start, args.range_end, args.range_interval,
        (args.round_packet or None), (args.round_burst or None)
    )

    y = np.array(labels)
    print(f"Loaded {len(files)} traces, {len(set(y))} classes")

    if 'vng' in attacks:
        mean,std = evaluate_vng_gaussiannb(X_vng,y,args.cv)
        print("VNG++ (GaussianNB) CV acc:", f"{mean:.4f} ± {std:.4f}")
        if args.out_model:
            clf = GaussianNB()
            clf.fit(X_vng, y)
            save_model(('vng', clf, {'range_start':args.range_start,'range_end':args.range_end,'range_interval':args.range_interval,'round_burst':args.round_burst}), args.out_model)

    if 'll-nb' in attacks:
        mean,std = evaluate_llnb_sets(X_llnb_sets, labels, args.cv)
        print("LL-NB (BernoulliNB) CV acc:", f"{mean:.4f} ± {std:.4f}")
        if args.out_model:
            mlb = MultiLabelBinarizer()
            X_bin = mlb.fit_transform(X_llnb_sets)
            clf = BernoulliNB()
            clf.fit(X_bin, labels)
            save_model(('ll-nb', (mlb, clf), {'round_packet': args.round_packet}), args.out_model)

    if 'jaccard' in attacks:
        mean,std = evaluate_jaccard(X_llnb_sets, labels, args.cv)
        print("LL-Jaccard CV acc:", f"{mean:.4f} ± {std:.4f}")

if __name__ == '__main__':
    main()
