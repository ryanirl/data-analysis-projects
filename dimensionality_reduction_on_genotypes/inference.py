import matplotlib.pyplot as plt
import seaborn as sb
import pandas as pd
import argparse


def main(args):
    df = pd.read_csv(args.vis_dir)

    sb.scatterplot(data = df, x = "PCA 1", y = "PCA 2", hue = "superpopulation_code")
    plt.show()

    sb.scatterplot(data = df, x = "TSNE 1", y = "TSNE 2", hue = "superpopulation_code")
    plt.show()

    sb.scatterplot(data = df, x = "UMAP 1", y = "UMAP 2", hue = "superpopulation_code")
    plt.show()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--vis_dir", type = str, default = "./data_vis.csv")

    args = parser.parse_args()
    
    main(args)



