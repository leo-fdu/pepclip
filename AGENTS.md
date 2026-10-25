# PepCLIP Project Guidelines

## Project Goals and Approach

- This project studies cross-modal representation alignment for short proteins and peptides: align peptide representations from a protein language model (PLM) with representations from a pretrained molecular graph model using CLIP-style contrastive learning.
- Training peptides must be **10–100 amino acids long, inclusive**. Data processing must document residue-counting rules, especially how noncanonical residues and modifications are handled.
- The baseline freezes both pretrained encoders and trains only the projection heads. Each modality uses a two-layer MLP (two linear layers with a nonlinear activation between them) to map representations into a shared space.
- Use paired representations of the same peptide molecule as positive pairs and a CLIP-style bidirectional contrastive loss. Specify normalization, temperature, negative-sample construction, and duplicate handling explicitly.
- The project will compare multiple combinations of PLMs and pretrained molecular graph models. Do not hard-code one model combination as the only option.
- Current decisions: the final encoder consists of the molecular graph encoder and its trained projection head. Training uses only natural peptide sequences (10–100 aa), initially filtered from the UniRef90 representative-sequence FASTA, with ESM2 and GROVER as the initial combination.
- Construct standardized linear peptide graphs from sequences using standard L-amino acids, free termini, and no additional disulfide bonds, cyclization, or modifications. Do not claim that these graphs represent the actual chemical structure of each sequence in vivo. A UniRef90 source does not imply experimental validation for every sequence.
- Do not crop long proteins by default. Splitting UniRef90 representative entries provides a baseline split, not a strict low-homology test. For stricter evaluation, supply external similarity groups and audit similarity across splits.
- The research goal is a peptide encoder that captures information at more levels and supports cyclic peptides and modified peptides containing noncanonical residues. **These are research goals that require validation, not demonstrated capabilities.**

## Scientific and Implementation Constraints

- Read the relevant code, configurations, and experiment records before making changes. Distinguish implemented behavior, existing experimental results, unvalidated hypotheses, and plans.
- Freezing an encoder must disable parameter gradient updates and explicitly set its runtime mode so that dropout or changes in running statistics do not undermine representation reproducibility. For cached representations, record the encoder version and preprocessing configuration used to produce the cache.
- Molecular graph inputs must preserve the chemical connectivity, cyclization, stereochemistry, and noncanonical residue information needed by the task. Specify how the PLM encodes nonstandard residues. Do not silently replace, discard, or linearize such information while claiming it is preserved.
- Alignment alone does not guarantee generalization to cyclic or modified peptides. Evaluation must specify the input modality, the encoder or projected representation used at inference, the supported molecular scope, and the training data's coverage of that scope.
- Record data sources and versions, cleaning rules, splitting methods, random seeds, model names and versions, pooling methods, projection dimensions, loss settings, and optimization settings.
- Prevent leakage across splits from identical molecules, duplicate sequences, and highly similar peptides. Specify how samples with the same sequence but different cyclization or modifications are paired and used as negatives.
- Keep data splits, training budgets, and evaluation procedures fixed across model combinations where possible, and compare against unaligned representations. If experimental semantics or comparison conditions change, explain the impact on comparability.
- Do not overwrite raw data or silently change splits or metrics. Use explicit configurations, separate data processing, encoding, projection, training, and evaluation logic, and keep changes minimal.

## Runtime Environment and Configuration

- **Project code runs on another machine; this machine is for development only. Local executability is not a prerequisite for implementation work.** Unless the user explicitly requests otherwise, do not install or repair runtime dependencies locally, configure a GPU environment, download model weights for execution, or change experimental implementations to accommodate this machine.
- Document and version-control the target runtime configuration: Python, PyTorch, CUDA, and related dependency versions; installation instructions and pinned versions or commits for upstream ESM/GROVER; weight acquisition instructions; path conventions; required environment variables; and commands for data processing, representation extraction, training, evaluation, and resuming training.
- If the target machine's hardware or software details are unavailable, expose the relevant settings as explicit parameters and mark unresolved items. Do not use this machine's macOS environment, existing Conda environments, or absolute paths as target runtime defaults.
- Missing local runtime dependencies are not a blocker. Do not set up additional local environments merely to run tests. Static checks that require no additional environment setup are allowed. For validation requiring real models, a GPU, or the target environment, provide explicit commands and clearly mark it as not executed. Do not equate local checks with target-environment validation.

## Git and Commit Rules

- **Commit every completed project change**, including code, configuration, documentation, and `.gitignore` changes. Each commit must represent a coherent, explainable change. Do not leave completed changes from the current task uncommitted at the end of the task.
- Before committing, inspect `git status`, the diff, and the staging area. Stage only explicit paths relevant to the current task. Never use `git add .` or indiscriminate staging.
- Preserve existing changes from the user or other tasks. Do not overwrite, revert, or include them in the current commit without authorization. Do not automatically push to a remote or rewrite history.
- Add checkpoints, model weights, `.DS_Store`, caches, runtime logs, environment directories, secrets, and other artifacts that should not be version-controlled to `.gitignore`. Do not commit them.
- `.gitignore` does not remove already tracked files. If a file should not be tracked, inspect the index and stop tracking it while retaining any locally needed copy. If sensitive information exists in history, report it explicitly and do not rewrite history without authorization.
- For other files, decide independently when the tracking policy is clear: source code, reproducible experiment configurations, dependency lists, documentation, and small test fixtures should generally be tracked; large datasets, reproducible intermediate artifacts, and machine-specific files generally should not.
- **If it is unclear whether a file should be tracked, ask the user to decide.** Do not add or remove it on your own or hide uncertain files behind overly broad ignore rules.
- Store large models and datasets externally. Keep the necessary acquisition instructions, versions, checksums, and metadata in the repository. Explicitly select any small result summaries or figures that need version control; do not commit entire output directories indiscriminately.

## Validation and Delivery

- After making changes, perform the smallest effective validation appropriate to the change. For documentation and ignore-rule changes, inspect the content, Git diff, and ignore behavior; no formal tests need to be added.
- For training-related changes, check freezing, tensor shapes, pairing, loss computation, and gradient flow as appropriate to the affected scope. State explicitly when full training has not been run.
- At delivery, briefly describe what changed, what was validated, what remains unverified, and the commit hash.
