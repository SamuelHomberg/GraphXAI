# %%
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

df1 = pd.read_csv("paper.csv", index_col=0)
df2 = pd.read_csv("gh.csv", index_col=0)
df3 = pd.read_csv("gh_weights.csv", index_col=0)
df4 = pd.read_csv("new_weights.csv", index_col=0)

df1["Data from"] = len(df1)*["paper"]
df2["Data from"] = len(df2)*["github (results)"]
df3["Data from"] = len(df3)*["github (model weights)"]
df4["Data from"] = len(df4)*["newly trained"]
df_big = pd.concat([df1, df2, df3, df4])


df_mutag = df_big[df_big["Dataset"]=='mutag'].reset_index(drop=True)
df_benzene = df_big[df_big["Dataset"]=='benzene'].reset_index(drop=True)
df_fc = df_big[df_big["Dataset"]=='fc'].reset_index(drop=True)
# %%
def make_bars(df, metric="GEA"):
    patches = plt.gca().patches
    if metric == "GEA":
        err = df['GEA_std'].dropna()
    if metric == "GEF":
        err = df['GEF_std'].dropna()
    for i,x in enumerate(patches):
        try:
            plt.errorbar(x.get_x()+x.get_width(), x.get_y()+x.get_height()/2,
                        xerr=err.iloc[i], color='k')
        except:
            pass # skip nan values

def plot(df, title=None, metric="GEA"):
    sns.barplot(df, x=metric, y="Method", hue="Data from", errorbar=None)
    make_bars(df)
    plt.title(title)
    return plt.gcf()

plot(df_mutag, "MUTAG", "GEA").savefig("compare_gea_mutag.png")
plt.close()
plot(df_benzene, "Benzene", "GEA").savefig("compare_gea_benzene.png")
plt.close()
plot(df_mutag, "Fluoride-Carbonyl", "GEA").savefig("compare_gea_fc.png")
plt.close()
plot(df_mutag, "MUTAG", "GEF").savefig("compare_gef_mutag.png")
plt.close()
plot(df_benzene, "Benzene", "GEF").savefig("compare_gef_benzene.png")
plt.close()
plot(df_mutag, "Fluoride-Carbonyl", "GEF").savefig("compare_gef_fc.png")
plt.close()
# %%
import numpy as np
def inspect(dataset="mutag", metric="GEA", results="results", method="RAND", node="node"):
    if metric.upper() == "GEA":
        metric_folder = "accuracy"
    if metric.upper() == "GEF":
        metric_folder = "faithfulness"
    path = f"{dataset}/{results}/{metric_folder}/{method.upper()}_{metric.upper()}_{node}.npy"
    return np.load(path)

inspect(metric="GEF", results="results_new_weights")