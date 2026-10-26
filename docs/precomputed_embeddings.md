# Precomputed embedding training

The main model sees two vector matrices, not sequences, molecular structures, or encoder implementations. Each side has its own two-layer MLP (`Linear → GELU → Linear`) followed by L2 normalization. Their input dimensions may differ; their projected dimensions must match. The optimizer contains only the two projection heads. External encoders are neither loaded nor executed by `train`, `evaluate`, or projection-only inference.

## Inputs and provenance

For each side provide:

- A plain NumPy `.npy` matrix `[N, D]`, with `D > 0`; float16/float32/float64 are accepted and rows are converted to FP32 for training. Object, complex, integer, nonfinite, and FP32-overflowing data are rejected. Ragged token/atom embeddings must first be pooled by the external extraction pipeline.
- `ids.json`: an array of unique nonempty strings, one per matrix row. A sample ID must refer to exactly the same positive-pair object on both sides. Row orders and dimensions may differ, and bundles may contain extra IDs outside a training selection; selected missing IDs fail immediately. The loader cannot establish molecular equivalence from vectors alone.
- A provenance JSON object with nonempty strings `encoder`, `version`, `pooling`, and `preprocessing`. Include weight SHA256, encoder source commit, dataset version, feature/graph policy, normalization, and upstream package versions as additional fields when applicable. Use explicit `unknown` values for unresolved information; do not infer versions from filenames.

Example provenance (replace placeholders):

```json
{
  "encoder": "YOUR_ENCODER_A",
  "version": "YOUR_PINNED_VERSION_OR_COMMIT",
  "pooling": "YOUR_EXACT_POOLING_RULE",
  "preprocessing": "YOUR_INPUT_AND_FEATURE_POLICY",
  "weights_sha256": "YOUR_WEIGHT_CHECKSUM",
  "source_dataset_version": "YOUR_DATA_RELEASE_OR_EXPLICIT_UNKNOWN"
}
```

Freeze external encoders with both `eval()` and disabled parameter gradients, extract without autograd, and record the exact extraction configuration. For PepCLIP's peptide experiments, continue enforcing full natural sequences of 10–100 canonical residues before extraction and the documented standardized linear peptide graph policy. Generic vectors carry no sequence/structure information from which the main trainer could independently verify those restrictions. Chemical identity, modifications, and stereochemistry must be handled correctly upstream; generic ingestion does not establish cyclic/noncanonical-peptide generalization.

Import the two externally computed matrices independently:

```bash
pepclip import-embeddings \
  --features /path/to/a.npy --ids /path/to/a-ids.json \
  --provenance /path/to/a-provenance.json --output embeddings/model-a

pepclip import-embeddings \
  --features /path/to/b.npy --ids /path/to/b-ids.json \
  --provenance /path/to/b-provenance.json --output embeddings/model-b
```

Each portable bundle contains `features.npy`, `ids.json`, and schema-2 `metadata.json`. Import **copies** the matrix, so budget space for the originals and copies; hashes are checked and completion metadata is written last. Existing output directories are refused. No pretrained model or encoder package is needed. Training memory-maps matrices and verifies full-bundle checksums/finite values even when selecting a subset, which adds startup disk I/O but avoids silently trusting modified caches.

## Pair selection and splits

Supply JSONL with one row per selected positive pair:

```jsonl
{"sample_id":"peptide-1","split":"train","identity_id":"molecule-1","group_id":"family-1"}
{"sample_id":"peptide-2","split":"train","identity_id":"molecule-2","group_id":"family-2"}
{"sample_id":"peptide-3","split":"val","identity_id":"molecule-3","group_id":"family-3"}
{"sample_id":"peptide-4","split":"val","identity_id":"molecule-4","group_id":"family-4"}
```

Only `sample_id` and `split` are required. `train` and `val` must each contain at least two pairs; `test` is optional. The trainer never creates or changes a split automatically. `identity_id` defaults to the sample ID and must be unique across the selection; use an upstream canonical molecular/sequence identity to detect duplicates across different IDs. `group_id` also defaults to the sample ID; supplied similarity groups cannot cross splits. Identical embeddings are not automatically treated as identical molecules.

For the initial canonical linear-peptide baseline, identical sequences correspond to the same standardized molecule and must be deduplicated before constructing pairs. If future data includes modifications/cyclization that a sequence encoder cannot distinguish, define pairing and negatives explicitly before using this single-positive loss; changing IDs alone does not resolve ambiguous positives. Sequence homology grouping/auditing remains an external data-curation responsibility.

Bundles are independent of pair selection and splitting, so a new subset or split can use the same embeddings without recomputation or copying. The selected pair manifest is separately fingerprinted in the training checkpoint. Resuming an existing run with changed splits, identities, groups, vectors, or source metadata is rejected; start a new run for those changes. Rows are sorted by sample ID to stabilize validation batch membership.

## Training, validation, and resumption

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8

pepclip train --config configs/precomputed.json \
  --pairs /path/to/pairs.jsonl \
  --embedding-a embeddings/model-a --embedding-b embeddings/model-b \
  --device cuda --output runs/a-b-h512

pepclip evaluate --checkpoint runs/a-b-h512/best.pt \
  --pairs /path/to/pairs.jsonl \
  --embedding-a embeddings/model-a --embedding-b embeddings/model-b \
  --split val --mode batch --device cuda

pepclip train --config configs/precomputed.json --epochs 40 \
  --pairs /path/to/pairs.jsonl \
  --embedding-a embeddings/model-a --embedding-b embeddings/model-b \
  --resume runs/a-b-h512/last.pt --device cuda --output runs/a-b-h512-continued
```

The supplied configuration uses FP32, hidden dimension 512, projection dimension 256, temperature 0.07, AdamW, training batch 8192, and independently fixed validation batch 8192. These are **unmeasured starting settings**, not a GPU capacity guarantee or an established optimum. CLI flags override the JSON configuration. No ESM2/GROVER names, registrations, weights, or chemistry packages are required.

Within each training/validation batch, matching IDs are the sole positives and all off-diagonal pairs are negatives. Loss is the average of A→B and B→A cross-entropies over temperature-scaled cosine similarities. Training shuffles deterministically by `seed + epoch`; validation uses fixed ID order. A final singleton is merged into the preceding batch (which may become `batch_size + 1`), and a split smaller than the batch size uses one smaller batch. Validation reports sample-weighted batch loss, pair count, batch count, and actual minimum/maximum batch sizes. Fix validation batch size and selection across comparisons because the negative pool changes the loss.

`best.pt` is selected by validation loss; `last.pt` stores the optimizer and RNG state for resumption. Training runs the configured total epoch count; automatic patience-based early stopping is not implemented. Resume only permits total epochs and device to change, and cross-device bitwise equivalence is not guaranteed.

Full-candidate retrieval is optional: add `--mode retrieval` to `evaluate`. Its A→B/B→A Recall@1/5/10, MRR, mean rank, and loss use every selected split sample as a candidate. This has quadratic time cost, differs from batch-local loss, and should not be substituted silently into a comparison. `--validation-mode retrieval` opts into this during training, but is unsuitable as a routine million-sample validation setting.

Legacy schema-1 peptide caches and CLI flags (`--manifest`, `--sequence-cache`, `--graph-cache`) remain supported. Their old whole-manifest binding still applies. Generic schema-2 bundles must be used on both sides; mixed schema versions are rejected. `--protein-encoder`/`--graph-encoder` are legacy-cache checks only and must be omitted for schema-2 inputs. Older checkpoints that predate batch validation can only resume while explicitly preserving their original full-retrieval validation and encoder configuration.

## Apply a trained head to new precomputed vectors

```python
from pepclip.inference import ProjectedEmbeddingEncoder

projector = ProjectedEmbeddingEncoder("runs/a-b-h512/best.pt", side="b", device="cpu")
z = projector.encode(new_b_embeddings)  # [N, original D_b] -> [N, projection_dim]
print(projector.source_metadata)        # Compare with the new vectors' extraction provenance.
```

This loads only the trained projection module. Changing the head does not require encoder recomputation. Changing encoder weights, preprocessing, pooling, or adding previously unseen samples requires the corresponding external extraction. The caller must ensure that new vectors come from the recorded encoder/preprocessing; dimension checks cannot prove provenance. In the peptide application, side B is conventionally the graph side, and actual molecular inference still requires its external graph encoder.

## Target environment and checks

Install only `pip install -e .` for the main module in the chosen target environment. Current declared bounds are Python >=3.10, PyTorch >=2.2, NumPy >=1.24; select and pin the exact Python/PyTorch/CUDA wheel, NVIDIA driver, and GPU pairing for each target before a real run rather than copying a development-machine environment. `provenance.json` records actual Python/PyTorch/NumPy/CUDA runtime and CUDA device, Git revision, and input provenance. No encoder weights, upstream ESM/GROVER checkouts, or encoder-specific environment variables are needed by this workflow.

A subsequently validated reference target is Python 3.10.12 / PyTorch 2.6.0+cu124 / NumPy 2.2.6 / driver 550.142 on RTX 4090. All six tensor tests passed there and synthetic projection-only GPU training was benchmarked; see the [measured results and reproduction settings](benchmarks/20261026-rtx4090-projection.md). This does not validate other GPU/software combinations, real encoder extraction, full-scale I/O, or downstream accuracy.

Run the synthetic tensor suite on that environment:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_pairs.py' -v
PYTHONPATH=src python -m unittest discover -s tests -p 'test_embeddings.py' -v
```

It checks input corruption and shape/dtype validation, arbitrary model names/different dimensions/reordered IDs, subset and split reuse, identity/group leakage, head-only gradients, weighted fixed-batch validation without full retrieval, CLI training/evaluation, exact CPU resumption, and projection inference. Static syntax/configuration/diff checks and six standard-library data tests (pair manifests and FASTA filtering) passed locally. These new tensor tests have **not been executed locally** because the development interpreter lacks PyTorch/NumPy; no dependencies were installed locally. Real model extraction, full-scale I/O throughput, other GPU configurations, and downstream performance remain unvalidated. Run the full `tests/` suite in an environment with the optional chemistry dependencies when validating the legacy extraction workflow.
