import pandas as pd
import numpy as np
import argparse
import warnings
import pickle

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from umap import UMAP

warnings.filterwarnings("ignore")


def main(args):
    df = pd.read_csv(args.data_dir)

    df_vis = df[["sample", "population_code", "superpopulation_code"]]
    df_snps = df.drop(["sample", "population_code", "superpopulation_code"], axis = 1)

    genotypes = df_snps.to_numpy()

    # Fit PCA Model
    if args.PCA:
        print("Fitting PCA")

        pca_model = PCA(n_components = 2)
        pca_model.fit(genotypes)
        pca_out = pca_model.transform(genotypes)

        df_vis["PCA 1"] = pd.to_numeric(pca_out[:, 0])
        df_vis["PCA 2"] = pd.to_numeric(pca_out[:, 1])

    # Fit t-SNE Model
    if args.tSNE:
        print("Fitting t-SNE")

        tsne_model = TSNE(n_components = 2)
        tsne_out = tsne_model.fit_transform(genotypes)

        df_vis["TSNE 1"] = pd.to_numeric(tsne_out[:, 0])
        df_vis["TSNE 2"] = pd.to_numeric(tsne_out[:, 1])

    # Fit UMAP Model
    if args.UMAP:
        print("Fitting UMAP")

        umap_model = UMAP(n_components = 2)
        umap_out = umap_model.fit_transform(genotypes)

        df_vis["UMAP 1"] = pd.to_numeric(umap_out[:, 0])
        df_vis["UMAP 2"] = pd.to_numeric(umap_out[:, 1])

    print("Saving 2D data projections to: ./vis_data.csv")
    df_vis.to_csv("./vis_data.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type = str, default = "./clean_data.csv")
    parser.add_argument("--PCA", type = bool, default = True)
    parser.add_argument("--tSNE", type = bool, default = True)
    parser.add_argument("--UMAP", type = bool, default = True)

    args = parser.parse_args()
    
    main(args)











