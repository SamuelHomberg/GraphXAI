# %%
import subprocess
import re
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt

def summarize_results(dataset:str, exp_method: str, metric: str):
    result = subprocess.run(['python', 'summarize_results.py',
                             '--dataset', dataset,
                             '--exp_method', exp_method,
                             '--metric', metric],
                             capture_output=True, text=True)
    return result.stdout

def parse_out(result: str):
    # match all floating point numbers
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", result)
    assert len(numbers) >= 2, f"unexpected result:\n{result}"
    # node_mean, node_std, (edge_mean, edge_std)
    return [float(x) for x in numbers]


def reconstruct_table(verbose = True):
    methods = ['RAND', 'GRAD', 'GCAM', 'GBP', 'IG', 'GNNEX', 'PGMEX', 'PGEX', 'SUBX']
    datasets = ['mutag', 'benzene', 'fc']
    metrics = ['GEA', 'GEF']

    data = {'Dataset': [], 'Method': [], 'GEA': [], 'GEA_std': [], 'GEF': [], 'GEF_std': []}
    
    for dataset in datasets:
        for method in methods:
            res_gea = summarize_results(dataset , method, 'GEA')
            res_gef = summarize_results(dataset , method, 'GEF')
            if res_gea:
                out_gea = parse_out(res_gea)
            else: 
                out_gea = [float('nan'), float('nan')]
            if res_gef:
                out_gef = parse_out(res_gef)
            else:
                out_gef = [float('nan'), float('nan')]
            if verbose:
                print(f"{dataset:<10}{method:<10}{out_gea[0]:>5.3f} +- {out_gea[1]:>5.3f}   {out_gef[0]:>5.3f} +- {out_gef[1]:>5.3f}")
            data['Dataset'] += [dataset]
            data['Method'] += [method]
            data['GEA'] += [out_gea[0]]
            data['GEA_std'] += [out_gea[1]]
            data['GEF'] += [out_gef[0]]
            data['GEF_std'] += [out_gef[1]]
    return pd.DataFrame(data)

def get_published_table():
    columns =  ['Dataset', 'Method', 'GEA', 'GEA_std', 'GEF', 'GEF_std']
    data = [
    ['mutag', 'RAND', 0.044, 0.007, 0.590, 0.031],
    ['mutag', 'GRAD', 0.022, 0.006, 0.598, 0.030],
    ['mutag', 'GCAM', 0.085, 0.012, 0.672, 0.029],
    ['mutag', 'GBP',  0.036, 0.007, 0.649, 0.030],
    ['mutag', 'IG',   0.049, 0.010, 0.443, 0.031],
    ['mutag', 'GNNEX',0.031, 0.005, 0.618, 0.030],
    ['mutag', 'PGMEX',0.042, 0.007, 0.503, 0.031],
    ['mutag', 'PGEX', 0.046, 0.007, 0.504, 0.031],
    ['mutag', 'SUBX', 0.039, 0.007, 0.611, 0.030],
    ['benzene', 'RAND', 0.108, 0.003, 0.513, 0.012],
    ['benzene', 'GRAD', 0.122, 0.007, 0.262, 0.011],
    ['benzene', 'GCAM', 0.291, 0.007, 0.551, 0.012],
    ['benzene', 'GBP',  0.205, 0.007, 0.438, 0.012],
    ['benzene', 'IG',   0.044, 0.003, 0.182, 0.010],
    ['benzene', 'GNNEX',0.129, 0.005, 0.444, 0.012],
    ['benzene', 'PGMEX',0.154, 0.006, 0.433, 0.012],
    ['benzene', 'PGEX', 0.169, 0.007, 0.375, 0.012],
    ['benzene', 'SUBX', 0.371, 0.009, 0.513, 0.012],
    ['fc', 'RAND', 0.087, 0.007, 0.440, 0.26],
    ['fc', 'GRAD', 0.132, 0.010, 0.210, 0.021],
    ['fc', 'GCAM', 0.005, 0.007, 0.500, 0.026],
    ['fc', 'GBP',  0.089, 0.010, 0.315, 0.024],
    ['fc', 'IG',   0.091, 0.007, 0.174, 0.019],
    ['fc', 'GNNEX',0.094, 0.009, 0.423, 0.026],
    ['fc', 'PGMEX',0.078, 0.008, 0.426, 0.026],
    ['fc', 'PGEX', 0.079, 0.009, 0.372, 0.025],
    ['fc', 'SUBX', 0.008, 0.002, 0.466, 0.026],
    ]
    published_df = pd.DataFrame(data, columns=columns)
    return published_df

def compare_dataframes(df1, df2, get_cbar=None|str):
    # run this in ipython with `# %%` for jupyter cells
    diff =   df1[['GEA', 'GEA_std', 'GEF', 'GEF_std']].round(3) \
           - df2[['GEA', 'GEA_std', 'GEF', 'GEF_std']].round(3)
    gmap = pd.concat([df1[['Dataset', 'Method']], diff], axis=1)
    vmax = max(abs(max(diff.max())), abs(min(diff.min())))
    styled_df1 = df1.style.background_gradient(cmap='PuOr', axis=None, gmap=gmap,
                                subset=['GEA', 'GEA_std', 'GEF', 'GEF_std'],
                                vmin=-vmax, vmax=vmax)
    if get_cbar is not None:
        norm = Normalize(vmin=-vmax, vmax=vmax)

        fig, ax = plt.subplots(figsize=(1, 3.5))
        fig.subplots_adjust(right=0.5)  # Make room for colorbar
        sm = ScalarMappable(norm=norm, cmap='PuOr')
        sm.set_array([])  # dummy
        cbar = fig.colorbar(sm, ax=ax)
        ax.remove()
        plt.show()
        # plt.savefig(get_cbar)
    return styled_df1

if __name__ == "__main__":
    df = reconstruct_table()
