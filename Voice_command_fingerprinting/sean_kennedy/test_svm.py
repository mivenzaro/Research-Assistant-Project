#!/usr/bin/env python3
import os
import csv
import argparse
import joblib
import numpy as np
from pathlib import Path
from sklearn.metrics import classification_report

def list_csv(dirpath):
    out=[]
    for root,_,files in os.walk(dirpath):
        for f in files:
            if f.lower().endswith(".csv"):
                out.append(os.path.join(root,f))
    out.sort()
    return out

def parse_rows(path):
    rows=[]
    with open(path,"r",encoding="utf-8",errors="ignore") as fh:
        reader=csv.reader(fh)
        try:
            first=next(reader)
        except StopIteration:
            return []
        hdr=[h.strip().lower() for h in first]
        if "size" in hdr and "direction" in hdr and "time" in hdr:
            ti=hdr.index("time"); si=hdr.index("size"); di=hdr.index("direction")
            for r in reader:
                if len(r)<=max(ti,si,di):
                    continue
                try:
                    t=float(r[ti]); s=int(r[si]); d=int(r[di]); rows.append((t,s,d))
                except:
                    continue
        else:
            fh.seek(0)
            for r in reader:
                if len(r)<4:
                    continue
                try:
                    t=float(r[1]); s=int(r[2]); d=int(r[3]); rows.append((t,s,d))
                except:
                    continue
    return rows

def bursts_from_rows(rows,gap_threshold=0.02):
    if not rows:
        return [],0,[]
    rows_sorted=sorted(rows,key=lambda x:x[0])
    bursts=[]
    cur=[rows_sorted[0]]
    for prev,cur_row in zip(rows_sorted,rows_sorted[1:]):
        if cur_row[0]-prev[0]>gap_threshold:
            bursts.append(cur)
            cur=[cur_row]
        else:
            cur.append(cur_row)
    bursts.append(cur)
    burst_sizes=[sum(abs(r[1]) for r in b) for b in bursts]
    burst_count=len(bursts)
    burst_durations=[b[-1][0]-b[0][0] for b in bursts]
    return burst_sizes,burst_count,burst_durations

def feature_stats(values):
    if not values:
        return [0.0]*6
    a=np.array(values,dtype=float)
    return [float(a.mean()),float(np.median(a)),float(a.std(ddof=0)),float(np.percentile(a,25)),float(np.percentile(a,75)),float(a.max())]

def extract_features(path):
    rows=parse_rows(path)
    if not rows:
        return None
    times=[r[0] for r in rows]
    sizes=[r[1] for r in rows]
    dirs=[r[2] for r in rows]
    incoming=[s for s,d in zip(sizes,dirs) if d<0]
    outgoing=[s for s,d in zip(sizes,dirs) if d>=0]
    total_in=sum(abs(s) for s in incoming)
    total_out=sum(abs(s) for s in outgoing)
    pkt_count=len(rows)
    in_ratio=(len(incoming)/pkt_count) if pkt_count else 0.0
    size_stats_all=feature_stats(sizes)
    size_stats_in=feature_stats(incoming)
    size_stats_out=feature_stats(outgoing)
    burst_sizes,burst_count,burst_durations=bursts_from_rows(rows)
    burst_stats=feature_stats(burst_sizes)
    avg_burst_dur=float(np.mean(burst_durations)) if burst_durations else 0.0
    inter=np.diff(sorted(times)) if len(times)>1 else np.array([0.0])
    ia_stats=[float(inter.mean()),float(np.median(inter)),float(inter.std(ddof=0))]
    feat=[total_in,total_out,pkt_count,in_ratio,burst_count,avg_burst_dur]
    feat+=size_stats_all+size_stats_in+size_stats_out+burst_stats+ia_stats
    return np.array(feat,dtype=float)

def infer_label(path):
    name=Path(path).stem
    if "_burst_" in name:
        name=name.split("_burst_")[0]
    if "_" in name:
        return name.split("_",1)[0]
    return Path(path).parent.name or name

def parse_skip(s):
    return set()

def main():
    p=argparse.ArgumentParser()
    p.add_argument("-m","--model",required=True)
    p.add_argument("-f","--file")
    p.add_argument("-d","--dir")
    p.add_argument("--normalize",action="store_true")
    a=p.parse_args()

    pkg=joblib.load(a.model)
    model=pkg.get("model")
    scaler=pkg.get("scaler",None)
    if model is None:
        raise SystemExit("model not found in package")

    if a.file:
        files=[a.file]
    elif a.dir:
        files=list_csv(a.dir)
    else:
        raise SystemExit("provide -f or -d")

    X=[]; paths=[]; true=[]
    for f in files:
        v=extract_features(f)
        if v is None:
            continue
        X.append(v); paths.append(f); true.append(infer_label(f))
    if not X:
        raise SystemExit("no valid test samples")
    X=np.vstack(X)
    if scaler is not None:
        X=scaler.transform(X)
    elif a.normalize:
        n=np.linalg.norm(X,axis=1,keepdims=True); n[n==0]=1.0; X=X/n

    preds=model.predict(X)
    probs=None
    try:
        probs=model.predict_proba(X)
    except:
        probs=None

    correct=sum(1 for t,p in zip(true,preds) if t==p)
    total=len(preds)
    print("Predictions:")
    for i,p in enumerate(paths):
        base=os.path.basename(p)
        if probs is not None:
            try:
                idx=list(model.classes_).index(preds[i]); pr=probs[i][idx]; print(f"  {base} -> pred:{preds[i]} true:{true[i]} ({pr:.3f})")
            except:
                print(f"  {base} -> pred:{preds[i]} true:{true[i]}")
        else:
            print(f"  {base} -> pred:{preds[i]} true:{true[i]}")
    acc=correct/total if total else 0.0
    print(f"\nAccuracy: {correct}/{total} = {acc:.4f}\n")
    print(classification_report(true,list(preds),zero_division=0))

if __name__=="__main__":
    main()
