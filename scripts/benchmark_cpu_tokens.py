"""CPU-only ESM2 token cache preparation; no model or weights are loaded."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import numpy as np
import torch
from esm.data import Alphabet

from pepclip.utils import file_sha256, write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--batch-size",type=int,default=32)
    parser.add_argument("--repeats",type=int,default=5)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    if min(args.batch_size,args.repeats)<1:raise ValueError("Sizes must be positive")
    torch.set_num_threads(1)
    alphabet=Alphabet.from_architecture("ESM-1b")
    convert=alphabet.get_batch_converter()
    rows=json.loads(args.sample.read_text())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    result=dict(sample_sha256=file_sha256(args.sample),sample_count=len(rows),
                batch_size=args.batch_size,repeats=[],alphabet_tokens=alphabet.all_toks,
                scope="tokenize + int64 NPZ write/fsync + exact roundtrip; CPU only, no model")
    for repeat in range(args.repeats):
        directory=args.output.parent/f"token-r{repeat}"
        directory.mkdir(exist_ok=False)
        size=0;start=time.perf_counter()
        for offset in range(0,len(rows),args.batch_size):
            selected=rows[offset:offset+args.batch_size]
            _,_,tokens=convert([(r["sample_id"],r["sequence"]) for r in selected])
            lengths=np.array([len(r["sequence"]) for r in selected],dtype=np.int64)
            path=directory/f"chunk-{offset:06d}.npz"
            with path.open("xb") as handle:
                np.savez_compressed(handle,tokens=tokens.numpy(),lengths=lengths)
                handle.flush();os.fsync(handle.fileno())
            with np.load(path,allow_pickle=False) as loaded:
                assert np.array_equal(tokens.numpy(),loaded["tokens"])
                assert np.array_equal(lengths,loaded["lengths"])
            size+=path.stat().st_size
        seconds=time.perf_counter()-start
        result["repeats"].append(dict(seconds=seconds,samples_per_second=len(rows)/seconds,
                                      compressed_bytes=size))
    result.update(status="complete",median_samples_per_second=statistics.median(
        r["samples_per_second"] for r in result["repeats"]))
    write_json(args.output,result)
    print(json.dumps(result,indent=2))


if __name__=="__main__":main()
