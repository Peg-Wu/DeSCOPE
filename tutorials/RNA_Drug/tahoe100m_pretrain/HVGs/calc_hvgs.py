import os
import scanpy as sc
import pickle as pkl
from tqdm.auto import tqdm

data_dir = "/fse/home/wupengpeng/perturbation_datasets/origin_datasets/Tahoe_100M"
h5ads = [os.path.join(data_dir, h5ad) for h5ad in os.listdir(data_dir)]

for h5ad in tqdm(h5ads, desc="Extract HVGs"):
    plate = h5ad.split("/")[-1].split("_")[0]
    adata = sc.read_h5ad(h5ad)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=5000)
    hvgs = adata.var_names[adata.var["highly_variable"]].tolist()
    with open(f"./{plate}_5k_hvgs.pkl", "wb") as f:
        pkl.dump(hvgs, f)