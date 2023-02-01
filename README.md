## [CIKM '22] Finding Heterophilic Neighbors via Confidence-based Subgraph Matching for Semi-supervised Node Classification

### Project Structure

```
.
├── README.md
├── requirements.txt
└── consm.py
```

### Setup

- Setup Conda, PyTorch, CUDA
  - > pip install -r requirements.txt
- Datasets will be downloaded automatically
  
### Usage

```bash
python3 ./consm.py [0~5], number means dataset
```
- 0: Cora, 1: Citeseer, 2: Pubmed, 3: Chameleon, 4: Squirrel, 5: Actor

### Experiments

- The node classification accuracy of best test / testing score with best validation will be shown
- Iteration will be 300 epochs
- For this version, the time complexity of subgraph matching module can be high (we'll update parallel computation later)

## Citation

```
https://dl.acm.org/doi/abs/10.1145/3511808.3557324
```
