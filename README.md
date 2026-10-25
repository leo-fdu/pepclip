# PepCLIP

Freeze a protein language model and a pretrained molecular graph model, then align their representations with bidirectional CLIP training using an independent `Linear → GELU → Linear` projection head for each modality. The final encoder is **GROVER + graph projection head + L2 normalization**. The initial combination is ESM2 / GROVER; encoder types, MLP hidden dimensions, and the shared embedding dimension are configurable.

The current implementation supports UniRef90 FASTA cleaning, sequence-to-linear-molecular-graph conversion, independent representation extraction, training from caches, bidirectional retrieval evaluation over the full candidate set, training resumption, and graph-side inference. Real pretrained weights and the complete UniRef90 dataset have not yet been validated in this project. Unit tests use explicitly labeled synthetic encoders and do not replace real-model validation.

## Installation

Python 3.10–3.12 is recommended. Training requires only PyTorch and NumPy; data preparation requires RDKit, and ESM2 extraction requires fair-esm.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[chem,esm]'
```

GROVER uses modules from the [official repository](https://github.com/tencent-ailab/grover) and trusted pretrained weights. Compatibility with its legacy dependency environment has not yet been resolved or validated in this project; a package with the same name is not installed automatically through pip dependencies. You can use a separate GROVER-compatible environment, expose this project's `src/` and the official repository through `PYTHONPATH`, generate standard caches, and return to the main environment for training. The adapter uses the atom fingerprint path of `GroverFpGeneration` and requires all of its parameters to load successfully. Pretraining task heads in the original checkpoint that are not part of the fingerprint model may be ignored.

## Data and Structure Conventions

- Read `.fasta` / `.fasta.gz` files and retain sequences of **10–100 aa** based on their actual length. Accept only `ACDEFGHIKLMNPQRSTVWY`; do not silently remove unknown residues or crop long proteins.
- Deduplicate identical sequences and retain mappings for duplicate entries in the rejection log. Entries with invalid lengths are counted only, to avoid generating an excessively large log for the full UniRef90 dataset.
- Generate linear L-peptides with RDKit `MolFromSequence(flavor=0)`, using free amine/carboxylic acid termini and no additional disulfide bonds, cyclization, or modifications. Use RDKit's default neutral form without physiological-pH protonation, and retain isomeric SMILES.
- The generated structure is a **standardized linear form**, not the natural mature molecular structure recorded in a database. A natural source in UniRef90 does not guarantee experimental evidence of existence.
- Hash-based splitting of UniRef90 representative sequences is stable and reproducible, but does not provide a strict low-homology split. For formal generalization evaluation, supply an external clustering-group CSV and audit leakage from similar sequences. Retrieval improvements cannot be directly interpreted as generalization to cyclic or modified peptides.
- Existing output directories are never overwritten. Partial outputs are retained after failures for diagnosis; use a new directory when rerunning.

## Workflow After Downloading the Data

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

Each epoch evaluates the entire validation set and reports bidirectional Recall@1/5/10, MRR, mean rank, and CLIP loss over the full candidate set. Similarity ties receive the worst rank. Chunked computation avoids allocating a full N×N matrix, but still takes O(N²) time. Fix an appropriate validation-set size before large-scale training rather than randomly resampling candidates each epoch. The test set is used only through an explicit `evaluate` command.

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

When resuming, only the target total epoch count and device may change; all other configuration, data, and encoder fingerprints must match. Bitwise equivalence across devices is not guaranteed. Continuous versus resumed training on CPU in the same environment is covered by a regression test. If deterministic CUDA matrix operations are needed, set `CUBLAS_WORKSPACE_CONFIG=:4096:8` before starting.

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

To add an encoder, implement the `FrozenEncoder` interface with `encode()`, `output_dim`, and `metadata()`, then register it through `register_encoder(modality, name, factory)`. The CLI currently exposes initialization options for ESM2/GROVER; models requiring additional parameters must also extend CLI argument routing. The cache-based trainer is independent of specific adapters.

## Validation

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

Coverage includes length boundaries, standard-residue filtering, structural composition and chirality, group-based splitting, pairing of reordered caches and corruption checks, freezing and gradient flow, contrastive loss, overfitting synthetic samples, full-candidate retrieval, and consistency when resuming training.

Validation with real resources is still pending for ESM2/GROVER checkpoint loading and extraction, compatibility with the official GROVER environment, processing time and scale for the full UniRef90 dataset, GPU execution, and actual peptide encoder performance. Database and pretrained-weight versions must be pinned and recorded for real runs.

Reference interfaces: [official ESM examples](https://github.com/facebookresearch/esm#usage-); [GROVER fingerprint](https://github.com/tencent-ailab/grover/blob/main/grover/model/models.py); [UniRef overview](https://www.ebi.ac.uk/training/online/courses/uniprot-quick-tour/the-uniprot-databases/uniref/).

Previously recorded local validation environment: Python 3.10.19, PyTorch 2.10.0, NumPy 2.2.6, and RDKit 2025.09.4; 14 unit tests passed. Adapter tests used stand-ins for upstream interfaces. Real ESM2/GROVER model validation remains pending until weights and dependency environments are available.
