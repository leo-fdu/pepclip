"""Target-environment equivalence checks and short real-model batch tuning."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

from pepclip.data import sequence_to_smiles


def compare_graph(rows):
    from argparse import Namespace
    import torch
    from grover.data.molgraph import MolGraph, BatchMolGraph
    from pepclip.encoders.fast_graph import fast_molgraph_class, pack_graph, unpack_graph
    from rdkit import RDLogger
    torch.set_num_threads(1)
    RDLogger.DisableLog("rdApp.warning")
    args = Namespace(bond_drop_rate=0)
    fast = fast_molgraph_class()
    smiles = [sequence_to_smiles(r["sequence"]) for r in rows]
    start = time.monotonic()
    original = BatchMolGraph([MolGraph(s, args) for s in smiles], args).get_components()
    original_seconds = time.monotonic()-start
    start = time.monotonic()
    optimized = BatchMolGraph([fast(s, args) for s in smiles], args).get_components()
    fast_seconds = time.monotonic()-start
    restored = unpack_graph(pack_graph(optimized))
    for expected, actual, decoded in zip(original, optimized, restored):
        if expected.dtype != actual.dtype or expected.shape != actual.shape or not torch.equal(expected, actual):
            raise ValueError("Optimized graph differs from original")
        if expected.dtype != decoded.dtype or expected.numpy().tobytes() != decoded.numpy().tobytes():
            raise ValueError("Packed graph is not bitwise identical")
    return dict(samples=len(rows), original_seconds=original_seconds, fast_seconds=fast_seconds)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--grover-source", type=Path, required=True)
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--model", choices=["graphs", "esm2", "grover"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=256)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--batches", type=int, nargs="+", default=[16,32,64,128])
    args = p.parse_args()
    sys.path.insert(0, str(args.grover_source.resolve()))
    for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[k] = "1"
    rows = json.loads(args.sample.read_text())[:args.samples]
    if args.model == "graphs":
        started = time.monotonic()
        with ProcessPoolExecutor(args.workers, mp_context=mp.get_context("spawn")) as pool:
            cases = list(pool.map(compare_graph, [rows[i:i+16] for i in range(0,len(rows),16)]))
        result = dict(model="graphs", samples=len(rows), graph_components_bitwise_equal=True,
                      packed_components_bitwise_equal=True, cases=cases, wall_seconds=time.monotonic()-started)
    else:
        import torch
        from torch.nn import functional as F
        from rdkit import RDLogger
        from pepclip.encoders.esm2 import ESM2Encoder
        from pepclip.encoders.grover import GroverEncoder
        from pepclip.encoders.fast_graph import fast_molgraph_class, pack_graph, unpack_graph
        torch.set_num_threads(1)
        RDLogger.DisableLog("rdApp.warning")
        torch.cuda.set_device(0)
        torch.manual_seed(42)
        encoder = (ESM2Encoder(model_name=args.checkpoint.stem, checkpoint=str(args.checkpoint), device="cuda")
                   if args.model == "esm2" else GroverEncoder(str(args.checkpoint), device="cuda", trusted_checkpoint=True))
        cases = []
        if args.model == "grover":
            from grover.data.molgraph import BatchMolGraph
            fast = fast_molgraph_class()
            smiles = [sequence_to_smiles(r["sequence"]) for r in rows]
            graphs = [fast(s, encoder.args) for s in smiles]
            original = BatchMolGraph([encoder.MolGraph(s, encoder.args) for s in smiles[:8]], encoder.args).get_components()
            restored = unpack_graph(pack_graph(BatchMolGraph(graphs[:8], encoder.args).get_components()))
            with torch.inference_mode():
                expected = encoder.model(original, [None]*8).cpu()
                actual = encoder.model(restored, [None]*8).cpu()
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for batch_size in args.batches:
            batches = []
            for i in range(0,len(rows),batch_size):
                selected = rows[i:i+batch_size]
                if args.model == "grover":
                    inputs = BatchMolGraph(graphs[i:i+batch_size], encoder.args).get_components()
                else:
                    _,_,inputs = encoder.converter([(r["sample_id"],r["sequence"]) for r in selected])
                batches.append((inputs, selected))
            def forward(inputs, selected):
                if args.model == "grover":
                    return encoder.model(inputs, [None]*len(selected)).detach().cpu().float()
                rep = encoder.model(inputs.cuda(), repr_layers=[encoder.layer], return_contacts=False)["representations"][encoder.layer]
                return torch.stack([rep[j,1:len(r["sequence"])+1].mean(0) for j,r in enumerate(selected)]).detach().cpu().float()
            try:
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
                    forward(*batches[0])
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    started = time.monotonic()
                    for bundle in batches:
                        output = forward(*bundle)
                        if not torch.isfinite(output).all():
                            raise ValueError("Nonfinite output")
                    torch.cuda.synchronize()
                    elapsed = time.monotonic()-started
                case = dict(batch_size=batch_size, samples_per_second=len(rows)/elapsed,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
                print(json.dumps(case), flush=True)
                cases.append(case)
            except torch.OutOfMemoryError:
                cases.append(dict(batch_size=batch_size,status="out_of_memory"))
                torch.cuda.empty_cache()
        # Same inputs and pooling: assess mixed precision separately from graph equivalence.
        if args.model == "esm2":
            check = rows[:8]
            _,_,inputs = encoder.converter([(r["sample_id"],r["sequence"]) for r in check])
        else:
            check,inputs = rows[:8],restored
        with torch.inference_mode():
            fp32 = forward(inputs,check)
            with torch.autocast("cuda",dtype=torch.float16):
                amp = forward(inputs,check)
        result = dict(model=args.model,metadata=encoder.metadata(),cases=cases,
            frozen=all(not p.requires_grad for p in encoder.model.parameters()),eval_mode=not encoder.model.training,
            graph_embedding_exact_equal=args.model=="grover",
            amp_min_cosine=F.cosine_similarity(fp32,amp).min().item(), amp_max_abs=(fp32-amp).abs().max().item(),
            torch=str(torch.__version__),cuda=torch.version.cuda)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("cases","metadata")}),flush=True)


if __name__ == "__main__":
    main()
