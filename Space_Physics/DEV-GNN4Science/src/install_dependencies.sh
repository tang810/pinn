#!/bin/bash

# Install PyTorch
pip install torch==2.1.0+cu118 -f https://download.pytorch.org/whl/torch_stable.html

# Install PyTorch Scatter
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0.html --quiet

# Install PyTorch Sparse
pip install torch-sparse -f https://data.pyg.org/whl/torch-2.1.0.html --quiet

# Install PyTorch Cluster
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.1.0.html --quiet

# Install PyTorch Geometric
pip install git+https://github.com/pyg-team/pytorch_geometric.git

# Install other packages
pip install -r requirements.txt
