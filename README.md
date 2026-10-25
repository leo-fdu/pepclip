# PepCLIP

冻结蛋白语言模型和分子图预训练模型，用两侧独立的 `Linear → GELU → Linear` 投影头做双向 CLIP 对齐。最终编码器为 **GROVER + 图侧投影头 + L2 归一化**。初始组合 ESM2 / GROVER；编码器类型、MLP 隐藏维度和共享空间维度均可配置。

当前实现支持 UniRef90 FASTA 清洗、序列生成线性分子图、独立表示提取、缓存训练、完整候选集双向检索评估、恢复训练和图侧推理。真实预训练权重和完整 UniRef90 数据尚未在此项目验证；单元测试使用明确标记的合成编码器，不代替真实模型验证。

## 安装

建议 Python 3.10–3.12。训练只依赖 PyTorch 和 NumPy；数据准备需要 RDKit，ESM2 提取需要 fair-esm。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[chem,esm]'
```

GROVER 使用 [官方仓库](https://github.com/tencent-ailab/grover) 的模块与可信预训练权重。其旧依赖环境尚未在本项目解决或验证，不在 pip 依赖中自动安装同名包。可以在独立的 GROVER 兼容环境中通过 `PYTHONPATH` 加载本项目 `src/` 和官方仓库，输出标准缓存后回到主环境训练。适配层使用 `GroverFpGeneration` 的 atom fingerprint 路径，并严格要求其全部参数加载成功，允许忽略原 checkpoint 中不属于 fingerprint 模型的预训练任务头。

## 数据与结构约定

- 读取 `.fasta` / `.fasta.gz`，按实际序列长度保留 **10–100 aa**，仅接受 `ACDEFGHIKLMNPQRSTVWY`，不静默删除未知残基，不截取长蛋白。
- 同序列去重，重复条目的映射保留在拒绝日志中。长度不合格条目仅计数，避免全量 UniRef90 生成巨型日志。
- RDKit `MolFromSequence(flavor=0)` 生成 L 型线性肽；游离胺／羧酸末端，无额外二硫键、环化和修饰。采用 RDKit 默认中性形式，不做生理 pH 质子化；保留异构 SMILES。
- 生成的是**标准化线性形式**，不是数据库记录的天然成熟分子结构。UniRef90 的“天然来源”不保证实验存在证据。
- UniRef90 代表序列哈希划分稳定、可复现，但不是严格低同源划分。正式泛化评估建议提供外部聚类分组 CSV，并审计近似序列泄漏；不能将检索提升直接解释为环肽／修饰肽泛化。
- 不覆盖已有输出目录。失败时保留部分输出供诊断，重跑请指定新的目录。

## 数据下载完成后的流程

以下路径和 release 为占位值；CLI 也可通过 `PYTHONPATH=src python -m pepclip.cli` 使用。

```bash
pepclip prepare \
  --fasta data/raw/uniref90.fasta.gz \
  --source-version YOUR_UNIREF_RELEASE \
  --output data/processed/uniref90-v1
```

输出 `manifest.jsonl`、`metadata.json`、`rejected.jsonl`。元数据记录源记录内容的流式 SHA256、release、RDKit 版本、生成图策略、过滤统计和划分种子。默认 train/val/test 约为 80/10/10；这是哈希分配比例，并非强制精确数量。可加 `--groups groups.csv`（字段 `sample_id,group_id`；必须覆盖所有保留样本），同组不会跨划分。

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

本地 ESM2 权重可通过 `--checkpoint` 指定；`--model` 应对应其型号，`--layer` 可指定表示层。ESM2 排除特殊 token 后做残基均值池化。GROVER 拼接 `atom_from_atom` 与 `atom_from_bond` 的分子内均值，不附加手工描述符。官方 GROVER checkpoint 序列化了 Python 对象，因此仅对可信来源使用 `--trusted-checkpoint`。官方实现只支持 CPU/CUDA 路径，当前适配层拒绝 MPS。

提取使用 `eval()`、冻结参数、`no_grad()`；失败立即中止，不静默丢弃单侧样本。缓存为磁盘映射 float32 `.npy`、ID 列表和完成元数据。元数据包含输入指纹、权重 SHA256、池化方式及维度；没有完成元数据的缓存不能用于训练。提取 batch size 与 CLIP batch size 独立，长肽图可能需要更小的提取批量。

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

命令行覆盖配置文件。温度默认固定 0.07；仅两个投影头进入优化器。不同编码器组合可以有不同输入维度，由缓存声明并校验，输出维度一致。batch 内以同一肽为唯一正样本，其余样本为负样本；最后单样本 batch 合并到前一批，不丢弃样本。缓存严格校验 ID、数据指纹、维度、有限值及文件校验值，并按 ID 配对而非行号。

每轮评估整个验证集，输出双向 Recall@1/5/10、MRR、平均排名和全候选集 CLIP loss。并列相似度按最差排名处理。分块计算避免分配完整 N×N 矩阵，但时间仍为 O(N²)；大规模正式训练前应固定合适的验证规模，不要在每轮任意随机抽样候选集。测试集只在显式 `evaluate` 时使用。

保存 `config.json`、`provenance.json`、`metrics.jsonl`、`last.pt`、`best.pt`。后两者分别为可恢复训练状态与最低验证 loss 的推理模型；不重复保存预训练编码器权重。

```bash
# 总目标训练轮数为 40，从 last.pt 的下一轮继续；使用新输出目录。
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

恢复时只允许改变目标总轮数和设备；其余配置、数据及编码器指纹必须一致。跨设备不承诺位级一致性；CPU 同环境的连续训练与恢复训练有回归测试。CUDA 确定性矩阵运算如需要，可在启动前设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`。

## 图侧推理与扩展

```python
from pepclip.encoders import build_encoder
from pepclip.inference import PeptideEncoder

backbone = build_encoder("graph", "grover", checkpoint="weights/grover_base.pt",
                         trusted_checkpoint=True, device="cuda")
encoder = PeptideEncoder("runs/esm2-grover-h512/best.pt", backbone, device="cuda")
z = encoder.encode_molecules(["YOUR_ISOMERIC_SMILES"])
```

输入完整分子结构，不自动从 SMILES 还原序列。代码可接收化学结构，但仅在线性天然肽上训练不证明其对环肽或非天然残基的效果。推理会核对编码器元数据与训练缓存是否一致。

新增编码器实现 `FrozenEncoder` 接口，提供 `encode()`、`output_dim`、`metadata()`，通过 `register_encoder(modality, name, factory)` 注册。CLI 当前提供 ESM2/GROVER 的初始化选项；新模型若需要额外参数，应同时扩展其 CLI 参数路由。缓存训练器不依赖具体适配器。

## 验证

```bash
PYTHONPATH=src python -m unittest discover -s tests -v
```

覆盖长度边界、标准残基过滤、结构组成与手性、分组划分、缓存乱序配对与损坏检查、冻结与梯度流、对比损失、合成样本过拟合、全候选检索及恢复训练一致性。

尚待真实资源验证：ESM2/GROVER checkpoint 加载与提取、官方 GROVER 环境兼容性、完整 UniRef90 处理时间与规模、GPU 路径、实际肽编码器效果。数据库和预训练权重版本必须在真实运行时固定并记录。

参考接口：[ESM 官方示例](https://github.com/facebookresearch/esm#usage-)；[GROVER fingerprint](https://github.com/tencent-ailab/grover/blob/main/grover/model/models.py)；[UniRef 说明](https://www.ebi.ac.uk/training/online/courses/uniprot-quick-tour/the-uniprot-databases/uniref/)。

本次本地验证环境：Python 3.10.19、PyTorch 2.10.0、NumPy 2.2.6、RDKit 2025.09.4；14 项 unittest 通过。适配层使用上游接口替身测试，ESM2/GROVER 的真实模型验证仍待权重与依赖环境就绪。
