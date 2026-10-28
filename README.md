# PepCLIP

The main training module accepts **any two sets of paired, precomputed embedding vectors** and trains only independent `Linear → GELU → Linear` projection heads using bidirectional CLIP loss. Input dimensions can differ. Training does not load or execute pretrained encoders and does not require RDKit, fair-esm, or GROVER. See the [precomputed embedding workflow](docs/precomputed_embeddings.md) for input formats, import, training, evaluation, resumption, and projection-only inference.

The research application aligns PLM and molecular-graph representations of peptides. Its final encoder is the selected molecular graph encoder plus its trained projection head and L2 normalization; ESM2/GROVER is the initial combination. UniRef90 preparation and encoder extraction remain separate optional tools. Sequence-only UniRef90 filtering has completed (see [run summary](docs/data/uniref90-sequences-20261026.json)); full-scale graph construction/extraction and downstream validation remain pending. Real ESM2/GROVER sample inference has been measured in the [encoder benchmark](docs/benchmarks/20261026-encoder-extraction.md). Synthetic tests do not replace real-model validation.

## Installation

Python 3.10–3.12 is recommended. Install the main module in the target runtime with `python -m pip install -e .`; it requires only PyTorch and NumPy. The optional peptide preparation/extraction tools additionally require RDKit/fair-esm as shown below. No local development environment setup is required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[chem,esm]'
```

GROVER uses modules from the [official repository](https://github.com/tencent-ailab/grover) and trusted pretrained weights. Sample inference has been validated on the pinned cloud environment in the encoder benchmark; compatibility with the original historical training environment remains unvalidated; a package with the same name is not installed automatically through pip dependencies. You can use a separate GROVER-compatible environment, expose this project's `src/` and the official repository through `PYTHONPATH`, generate standard caches, and return to the main environment for training. The adapter uses the atom fingerprint path of `GroverFpGeneration` and requires every learned parameter and supplied required tensor shape to match. Encoder-only checkpoints may omit dropout (defaulted to zero for eval) and the nonlearned empty-molecule zero sentinel; these exceptions are recorded in cache metadata. Pretraining task heads in the original checkpoint that are not part of the fingerprint model may be ignored.

## Data and Structure Conventions

- Read `.fasta` / `.fasta.gz` files and retain sequences of **10–100 aa** based on their actual length. Accept only `ACDEFGHIKLMNPQRSTVWY`; do not silently remove unknown residues or crop long proteins.
- Deduplicate identical sequences and retain mappings for duplicate entries in the rejection log. Entries with invalid lengths are counted only, to avoid generating an excessively large log for the full UniRef90 dataset.
- Generate linear L-peptides with RDKit `MolFromSequence(flavor=0)`, using free amine/carboxylic acid termini and no additional disulfide bonds, cyclization, or modifications. Use RDKit's default neutral form without physiological-pH protonation, and retain isomeric SMILES.
- The generated structure is a **standardized linear form**, not the natural mature molecular structure recorded in a database. A natural source in UniRef90 does not guarantee experimental evidence of existence.
- Hash-based splitting of UniRef90 representative sequences is stable and reproducible, but does not provide a strict low-homology split. For formal generalization evaluation, supply an external clustering-group CSV and audit leakage from similar sequences. Retrieval improvements cannot be directly interpreted as generalization to cyclic or modified peptides.
- Existing output directories are never overwritten. Partial outputs are retained after failures for diagnosis; use a new directory when rerunning.

## Workflow After Downloading the Data

For the user-requested CPU-first schedule, [local M5 Pro measurements](docs/benchmarks/20261026-m5pro-cpu-preprocessing.md) estimate roughly 22.5 hours of full-set graph/token preparation and about 242 GB of prepared caches before margins. The separate [4090 measurements](docs/benchmarks/20261026-encoder-extraction.md) estimate about 27.3 hours of GPU-stage work with inputs prepared. These are sampled extrapolations; the full sharded producer/consumer and resumption flow has not yet been implemented or run.

For a sequence-only first pass on a large release, use the standard-library-only command below. It streams compressed input without unpacking it, retains complete canonical 10–100 aa sequences, and deduplicates exact sequences using a temporary disk-backed SQLite index (64 MiB page cache). No model dependencies, molecular graphs, or train/validation/test assignments are required at this stage.

```bash
PYTHONPATH=src python -m pepclip.filter_fasta \
  --fasta /path/to/uniref90.fasta.gz \
  --source-version YOUR_UNIREF_RELEASE \
  --output data/processed/uniref90-sequences-v1
```

Outputs are `peptides.fasta` (first original header retained), `rejected.jsonl` (nonstandard residues and duplicate mappings), `progress.json` (every million input records), and `metadata.json` (written only on successful completion). Metadata includes full-input record-content SHA256, source file size/mtime, filtering counts, and retained length histogram. Length is counted after concatenating FASTA sequence lines and trimming line-edge whitespace; only uppercase standard residue letters are accepted, with no cropping, substitution, or modification interpretation. Rejection precedence is length, nonstandard residue, then duplicate sequence; invalid-residue statistics cover only the length-eligible records. Use an explicit `unknown` source version if the downloaded release cannot be established; file dates alone do not establish the release. Existing output directories are refused and failed partial outputs remain for diagnosis. Feed `peptides.fasta` to `prepare` below for graph construction and splitting. Its metadata will describe the filtered input; retain the first-pass metadata for full-source provenance. Hash-based split assignment for each retained ID remains the same.

The paths and release below are placeholders. The CLI can also be invoked with `PYTHONPATH=src python -m pepclip.cli`.

```bash
pepclip prepare \
  --fasta data/raw/uniref90.fasta.gz \
  --source-version YOUR_UNIREF_RELEASE \
  --output data/processed/uniref90-v1
```

This produces `manifest.jsonl`, `metadata.json`, and `rejected.jsonl`. Metadata records a streaming SHA256 of the source record content, the release, RDKit version, graph construction policy, filtering statistics, and split seed. The default train/validation/test proportions are approximately 80/10/10; these are hash-assignment proportions, not enforced exact counts. Add `--groups groups.csv` (columns: `sample_id,group_id`; all retained samples must be covered) to keep each group within one split.

```bash
pepclip extract \
  --manifest data/processed/uniref90-v1/manifest.jsonl \
  --modality sequence --encoder esm2 --model esm2_t33_650M_UR50D \
  --batch-size 16 --device cuda \
  --output embeddings/esm2-650m

pepclip extract \
  --manifest data/processed/uniref90-v1/manifest.jsonl \
  --modality graph --encoder grover \
  --checkpoint weights/grover_base.pt --trusted-checkpoint \
  --batch-size 8 --device cuda \
  --output embeddings/grover-base
```

Use `--checkpoint` to specify local ESM2 weights; `--model` must match the model variant, and `--layer` can select the representation layer. ESM2 uses mean pooling over residues after excluding special tokens. GROVER concatenates the within-molecule means of `atom_from_atom` and `atom_from_bond`, without adding handcrafted descriptors. Official GROVER checkpoints contain serialized Python objects, so use `--trusted-checkpoint` only for trusted sources. The official implementation supports CPU/CUDA paths only; the current adapter rejects MPS.

Extraction uses `eval()`, frozen parameters, and `no_grad()`. Failures abort immediately rather than silently dropping samples from one modality. Each cache contains a memory-mapped float32 `.npy` array, an ID list, and completion metadata. Metadata includes the input fingerprint, weight SHA256, pooling method, and dimensions; caches without completion metadata cannot be used for training. Extraction batch size is independent of CLIP batch size; long peptide graphs may require smaller extraction batches.

```bash
pepclip train \
  --config configs/esm2_grover.json \
  --manifest data/processed/uniref90-v1/manifest.jsonl \
  --sequence-cache embeddings/esm2-650m \
  --graph-cache embeddings/grover-base \
  --protein-encoder esm2 --graph-encoder grover \
  --hidden-dim 512 --projection-dim 256 \
  --device cuda --output runs/esm2-grover-h512
```

Command-line arguments override the configuration file. Temperature is fixed at 0.07 by default; only the two projection heads are included in the optimizer. Different encoder combinations may have different input dimensions, declared and validated by their caches, but must share the same output dimension. Within a batch, the matching peptide is the sole positive and all other samples are negatives. A final single-sample batch is merged into the preceding batch rather than discarded. Caches are strictly checked for IDs, data fingerprints, dimensions, finite values, and file checksums, and are paired by ID rather than row number.

By default, each epoch computes sample-weighted **batch-local** bidirectional CLIP loss over the validation split, with independently configurable `validation_batch_size`. Validation order is fixed and a final singleton is merged into the previous batch. Full-candidate retrieval is opt-in (`evaluate --mode retrieval`, or `--validation-mode retrieval` during training), costs O(N²) time, and reports bidirectional Recall@1/5/10, MRR, mean rank, and retrieval loss; ties receive the worst rank. Batch loss and full-candidate retrieval loss have different negative pools and are not interchangeable. Training saves the best checkpoint but still runs the configured number of epochs; automatic patience-based early stopping is not implemented. A test split is optional for generic bundles and is used only through an explicit `evaluate --split test`.

Training saves `config.json`, `provenance.json`, `metrics.jsonl`, `last.pt`, and `best.pt`. The last two contain resumable training state and the inference model with the lowest validation loss, respectively. Pretrained encoder weights are not duplicated.

```bash
# Target 40 total epochs, resuming at the epoch after last.pt; use a new output directory.
pepclip train \
  --config configs/esm2_grover.json --epochs 40 --device cuda \
  --manifest data/processed/uniref90-v1/manifest.jsonl \
  --sequence-cache embeddings/esm2-650m --graph-cache embeddings/grover-base \
  --resume runs/esm2-grover-h512/last.pt --output runs/esm2-grover-h512-continued

pepclip evaluate \
  --checkpoint runs/esm2-grover-h512/best.pt --split test --device cuda \
  --manifest data/processed/uniref90-v1/manifest.jsonl \
  --sequence-cache embeddings/esm2-650m --graph-cache embeddings/grover-base
```

When resuming, only the target total epoch count and device may change; all other configuration, data, and encoder fingerprints must match. Checkpoints from before batch-local validation require explicitly preserving their old settings with `--validation-mode retrieval --validation-batch-size ORIGINAL_BATCH_SIZE` and the original encoder-name configuration. Bitwise equivalence across devices is not guaranteed. Continuous versus resumed training on CPU in the same environment is covered by a regression test. For deterministic CUDA matrix operations, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting.

## Graph-Side Inference and Extensions

```python
from pepclip.encoders import build_encoder
from pepclip.inference import PeptideEncoder

backbone = build_encoder("graph", "grover", checkpoint="weights/grover_base.pt",
                         trusted_checkpoint=True, device="cuda")
encoder = PeptideEncoder("runs/esm2-grover-h512/best.pt", backbone, device="cuda")
z = encoder.encode_molecules(["YOUR_ISOMERIC_SMILES"])
```

Provide complete molecular structures; sequences are not automatically reconstructed from SMILES. The code can accept chemical structures, but training only on linear natural peptides does not demonstrate effectiveness on cyclic peptides or noncanonical residues. Inference checks that encoder metadata matches the training cache.

Externally computed embeddings require **no encoder registration or adapter changes**: import them into generic bundles. Only to extend the optional `extract` tool, implement the `FrozenEncoder` interface and register it through `register_encoder(modality, name, factory)`; extraction models requiring additional parameters must also extend CLI routing. The composed `PeptideEncoder` above uses legacy graph-cache metadata; use `ProjectedEmbeddingEncoder` with generic bundles and externally computed features.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Coverage includes length boundaries, standard-residue filtering, structural composition and chirality, group-based splitting, pairing of reordered caches and corruption checks, freezing and gradient flow, contrastive loss, overfitting synthetic samples, full-candidate retrieval, and consistency when resuming training.

The generic bundle/training regression suite can run without chemical or backbone packages: `PYTHONPATH=src python -m unittest discover -s tests -p 'test_embeddings.py' -v`. Together with `tests/test_pairs.py`, it covers different vector dimensions, reordered IDs, reusable subsets/splits, rejected corruption/nonfinite values, duplicate identity/group leakage, projection gradients, batch validation, resumption, and projection-only inference. Local syntax/configuration/diff checks and six standard-library data tests passed. Tensor tests were subsequently run in the cloud environment described below. The later user-requested CPU preprocessing benchmark uses a separate local environment. Real ESM2/GROVER loading and sampled GPU inference have since been validated; full-scale extraction and downstream performance remain unvalidated. Database, weight, and exact target runtime versions must be pinned and recorded for real runs.

Reference interfaces: [official ESM examples](https://github.com/facebookresearch/esm#usage-); [GROVER fingerprint](https://github.com/tencent-ailab/grover/blob/main/grover/model/models.py); [UniRef overview](https://www.ebi.ac.uk/training/online/courses/uniprot-quick-tour/the-uniprot-databases/uniref/).

Subsequent cloud validation: all six generic tensor tests passed with Python 3.10.12, PyTorch 2.6.0+cu124, and NumPy 2.2.6. Projection-only CUDA training was measured on an RTX 4090 across batches 2048–32768; see the [benchmark report](docs/benchmarks/20261026-rtx4090-projection.md). These synthetic warm-cache measurements do not validate pretrained encoder extraction or full-data I/O.

Previously recorded local validation environment: Python 3.10.19, PyTorch 2.10.0, NumPy 2.2.6, and RDKit 2025.09.4; 14 unit tests passed. Adapter tests used stand-ins for upstream interfaces. These historical adapter tests used stand-ins; the separate real-encoder benchmark documents subsequent sample inference validation.

### Independent projection widths and low-memory training

`hidden_dim_a` and `hidden_dim_b` independently set the two MLP hidden widths
(A = sequence, B = graph in the ESM/GROVER setup). A value of `0` inherits the
legacy `hidden_dim`; old configs and checkpoints remain readable. Input dimensions
are inferred from the bundles; `projection_dim` is shared. Both heads still use
Linear → GELU → Linear and L2 normalization. CLI overrides are
`--hidden-dim-a`, `--hidden-dim-b`, and `--projection-dim`.

For the explicitly requested local MPS experiment, use:

```bash
pepclip train --pairs PATH/pairs.jsonl \
  --embedding-a PATH/sequence_bundle --embedding-b PATH/graph_bundle \
  --output outputs/mps_low_memory \
  --config configs/precomputed_mps_low_memory.json
```

This opt-in config selects 8192/7200 hidden widths, 2048 output dimensions,
FP32 training, batch 256 for training and validation, and `low_memory: 1`.
For 2560/1600 input vectors it implements the requested architectures. The
existing target-machine configs are unchanged. No backbone is loaded or trained.

Low-memory mode currently requires **schema-2 bundles** and C-order `.npy`
features. It streams the existing JSONL pair manifest and JSON ID arrays into a
temporary SQLite index, enforcing duplicate/identity/group-split rules and
checking bundle hashes and finite values. It preserves the normal manifest
fingerprint, ID pairing, and sorted validation order. Indexes are rebuilt at
startup (including resume); checks scan all input files before training. Allow
startup time and disk space for the index. Set `index_dir` / `--index-dir` to an
**existing disk directory** with ample free space, not a RAM filesystem; empty
uses the system temporary directory. Source files are never overwritten. Indexes
are removed when the input objects are released normally; after a process kill,
unused `pepclip-index-*` temporary directories may require manual removal.

SQLite uses an 8 MiB page cache and disk temporary storage. Feature validation
reads approximately 4 MiB chunks; training reads individual rows into FP32
without keeping the feature files mapped. The DataLoader uses zero workers,
with no worker copies, pinned-memory queue or background prefetch. This trades
I/O speed for controlled application memory. RAM usage for metadata no longer
scales as a Python dict/list per dataset entry, though on-disk index size and
initialization time still grow with dataset size.

`shuffle_buffer` (default 65536 indices) bounds shuffle memory in this mode.
A seeded replacement buffer shuffles the sorted training stream, covers every
training pair exactly once per epoch, and merges a final singleton. This is
**not a uniform global permutation**: batch membership and negatives differ
from the original sampler, and nearby source IDs may remain correlated. Keep
this setting fixed across model comparisons. Do not interpret it as an exact
reproduction of an earlier global-shuffle run. Resume requires the same mode,
buffer size, widths and batches; only the index location may change. The normal
sampler retains its global shuffle but no longer builds all Python batch lists.

Only `validation_mode: "batch"` is allowed with low-memory inputs, including
standalone evaluation: full retrieval accumulates all projections and has
quadratic comparison work. Checkpoints remain once per completed epoch; an
interruption during an epoch restarts that epoch. There is no automatic OOM
retry or batch reduction, since that would silently change CLIP negatives.

This reduces dataset-size-dependent RAM growth; it is **not an unconditional
OOM guarantee**. GPU usage still depends on batch, widths, output dimension,
optimizer state, and other processes sharing memory. Start with the supplied
batch 256, close memory-heavy applications, and verify on real inputs before
long runs. If necessary choose a smaller batch explicitly and start a new run;
record the changed negative count. FP16 storage is supported and converted to
FP32 per row (10 million 2560+1600 vectors: about 83.2 GB FP16 or 166.4 GB FP32,
excluding metadata/indexes/checkpoints). Synthetic CPU regression checks do not
constitute a ten-million-sample or sustained MPS validation.

```bash
PYTHONPATH=src python -m unittest discover -s tests -p 'test_low_memory.py'
PYTHONPATH=src python -m unittest discover -s tests -p 'test_embeddings.py'
```
