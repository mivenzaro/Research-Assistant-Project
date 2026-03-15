#!/usr/bin/env python3
import os
import csv
import argparse
import joblib
import numpy as np
import re
from pathlib import Path
from sklearn.naive_bayes import GaussianNB

def round_size(size,b):
    return ((size+b//2)//b)*b

def infer_label(path):
    stem=Path(path).stem
    if "_burst_" in stem:
        stem=stem.split("_burst_")[0]
    stem=re.sub(r'_\d+_\d+s$','',stem)
    stem=re.sub(r'_\d+s$','',stem)
    return stem

def read_trace(path,bin_size,skip):
    vals=[]
    with open(path,"r",encoding="utf-8",errors="ignore") as f:
        first=f.readline()
        if not first:
            return vals
        header=[h.strip().lower() for h in first.strip().split(',')]
        size_idx=None
        dir_idx=None
        for i,h in enumerate(header):
            h=h.replace(' ','').replace('_','')
            if h in ('size','packetsize','packetsizebytes','length'):
                size_idx=i
            if h in ('direction','dir'):
                dir_idx=i
        f.seek(0)
        r=csv.reader(f)
        if size_idx is not None and dir_idx is not None:
            next(r,None)
        for row in r:
            if not row:
                continue
            try:
                if size_idx is not None and dir_idx is not None:
                    s=int(row[size_idx])
                    d=int(row[dir_idx])
                else:
                    s=int(row[2])
                    d=int(row[3])
                if s in skip:
                    continue
                rs=round_size(s,bin_size)
                if d<0:
                    rs=-rs
                vals.append(rs)
            except:
                continue
    return vals

def make_bins(start,end,step):
    bins=[]
    v=start
    while v<end:
        bins.append(v)
        v+=step
    m={}
    n=len(bins)
    for i,b in enumerate(bins):
        m[-b]=i
        m[b]=i+n
    return m,n*2

def feature_vec(values,m,size):
    v=np.zeros(size)
    for x in values:
        if x in m:
            v[m[x]]+=1
    return v

def collect_files(d):
    out=[]
    for root,_,f in os.walk(d):
        for x in f:
            if x.endswith(".csv"):
                out.append(os.path.join(root,x))
    out.sort()
    return out

def parse_skip(s):
    out=set()
    if not s:
        return out
    for x in s.split(","):
        try:
            out.add(int(x))
        except:
            pass
    return out

def main():
    p=argparse.ArgumentParser()
    p.add_argument("-i","--input",required=True)
    p.add_argument("-o","--model",required=True)
    p.add_argument("--start",type=int,default=0)
    p.add_argument("--end",type=int,default=1500)
    p.add_argument("--bin",type=int,default=50)
    p.add_argument("--skip",default="")
    p.add_argument("--normalize",action="store_true")
    a=p.parse_args()

    skip=parse_skip(a.skip)
    files=collect_files(a.input)
    m,flen=make_bins(a.start,a.end,a.bin)

    X=[]
    y=[]
    for f in files:
        vals=read_trace(f,a.bin,skip)
        if not vals:
            continue
        X.append(feature_vec(vals,m,flen))
        y.append(infer_label(f))

    X=np.vstack(X)
    y=np.array(y)

    if a.normalize:
        n=np.linalg.norm(X,axis=1,keepdims=True)
        n[n==0]=1
        X=X/n

    clf=GaussianNB()
    clf.fit(X,y)

    pkg={"model":clf,"map":m,"flen":flen,"bin":a.bin}
    joblib.dump(pkg,a.model)
    print("saved model:",a.model)

if __name__=="__main__":
    main()
