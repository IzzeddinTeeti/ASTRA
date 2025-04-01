# Build environment for ASTRA model
pip install -r requirements.txt
pip install torch-sparse torch-scatter -f https://data.pyg.org/whl/torch-2.0.0+cu117.html --no-cache-dir
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.0.0+cu117.html