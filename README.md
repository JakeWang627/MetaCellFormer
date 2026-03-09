
# MetaCellFormer

MetaCellFormer is a deep learning framework for **cross-species single-cell integration** that aligns cells across species by learning a shared **metagene representation** and refining cell embeddings with a **Transformer-based metric learning framework**.

The model integrates multimodal gene embeddings (protein sequence embeddings and gene-text embeddings) to construct **metagenes**, enabling cross-species comparison beyond strict homologous gene mappings.

---

## Overview

Cross-species integration of single-cell transcriptomic data is challenging due to large gene heterogeneity between species. Many existing methods rely on **one-to-one homologous genes**, which often represent less than 30% of the total gene set and may discard species-specific information.

MetaCellFormer addresses this limitation by:

1. Mapping heterogeneous genes from different species into a **shared metagene space**
2. Learning **cell embeddings in metagene space**
3. Aligning cells across species using **Transformer-based attention and metric learning**

This framework enables robust alignment of the **same cell types across species** while maintaining separation between different cell types.

---

## Model Architecture

MetaCellFormer consists of two major stages:

### Stage 1: Metagene Representation Learning

Genes from multiple species are projected into a shared **metagene space** using multimodal gene embeddings.

Two types of gene embeddings are used:

- **Protein embeddings** derived from protein language models (e.g. ESM2)
- **Gene text embeddings** derived from gene functional descriptions

These embeddings are fused to construct **metagenes**, which represent groups of functionally related genes.

An autoencoder then projects cell expression profiles into this metagene space to produce **initial cell embeddings**.

---

### Stage 2: Cross-Species Cell Alignment

After obtaining metagene-level embeddings, MetaCellFormer refines cell representations using:

- **Transformer self-attention** to model global cell–cell relationships
- **Graph neural networks (GNN)** to propagate information between cells of the same type
- **Metric learning (SoftTriple loss)** to align cells of the same type across species

Three loss functions are jointly optimized:

- **SoftTriple loss** — Aligns cells of the same type across species
- **Intra-class variance loss** — Encourages compact clustering of cells with the same cell type
- **MMD loss** — Reduces species-specific batch effects within each cell type

---

## Training Pipeline

MetaCellFormer training is performed in **two steps**.

### Step 1: Metagene Projection using SATURN module

```bash
python train-saturn.py
```

This step maps genes from different species into a **shared metagene space** and generates initial cell embeddings.

### Step 2: Cross-Species Cell Alignment

```bash
python MetaCellFormer.py
```

This step aligns cells across species so that:

- same-type cells across species are close
- different cell types remain separated

---

## Applications

- Cross-species cell-type alignment
- Cross-species label transfer
- Multispecies cell atlas construction
- Metagene differential analysis
- Cell-type reannotation

---

## Requirements

- Python >= 3.9
- PyTorch
- Scanpy
- NumPy
- SciPy
- scikit-learn

---

## Citation

MetaCellFormer: Multi-Source Metagene Embedding with Transformer-Based Metric Learning for Cross-Species Single-Cell Integration

---

## Contact

Tianxu Wang
