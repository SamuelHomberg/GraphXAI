# %%
import numpy as np
print("same model, same seed")
print("GEA:")
for method in ["GCAM", "RAND"]:
    print(f"{method}")
    means = []
    for seed in ["912", "912_1", "912_2", "912_3", "912_4"]:
        path_name = f"mutag/results_{seed}/accuracy/{method}_GEA_node.npy"
        results = np.load(path_name)
        print(f"{seed:>15}: {np.mean(results):>8.4f} {np.std(results):.4f}")
        means.append(np.mean(results))
    print(f"all: {np.mean(means):>8.4f} {np.std(means):.4f}")
    print()

print("---\nGEF:")
for method in ["GCAM", "RAND"]:
    print(f"{method}")
    means = []
    for seed in ["912", "912_1", "912_2", "912_3", "912_4"]:
        path_name = f"mutag/results_{seed}/faithfulness/{method}_GEF_node.npy"
        results = np.load(path_name)
        print(f"{seed:>15}: {np.nanmean(results):>8.4f} {np.nanstd(results):.4f}")
        means.append(np.nanmean(results))
    print(f"all: {np.mean(means):>8.4f} {np.std(means):.4f}")
    print()

print("same model, different seed")
print("GEA:")
for method in ["GCAM", "RAND"]:
    print(f"{method}")
    means = []
    for seed in [912, 913, 914, 915, 916]:
        path_name = f"mutag/results_{seed}/accuracy/{method}_GEA_node.npy"
        results = np.load(path_name)
        print(f"{seed:>15}: {np.mean(results):>8.4f} {np.std(results):.4f}")
        means.append(np.mean(results))
    print(f"all: {np.mean(means):>8.4f} {np.std(means):.4f}")
    print()

print("---\nGEF:")
for method in ["GCAM", "RAND"]:
    print(f"{method}")
    means = []
    for seed in [912, 913, 914, 915, 916]:
        path_name = f"mutag/results_{seed}/faithfulness/{method}_GEF_node.npy"
        results = np.load(path_name)
        print(f"{seed:>15}: {np.nanmean(results):>8.4f} {np.nanstd(results):.4f}")
        means.append(np.nanmean(results))
    print(f"all: {np.mean(means):>8.4f} {np.std(means):.4f}")
    print()

print("different trained models")
print("GEA:")
for method in ["GCAM", "RAND"]:
    print(f"{method}")
    means = []
    for seed in ["912_model_1", "912_model_2", "912_model_3", "new_weights", "912"]:
        path_name = f"mutag/results_{seed}/accuracy/{method}_GEA_node.npy"
        results = np.load(path_name)
        print(f"{seed:>15}: {np.mean(results):>8.4f} {np.std(results):.4f}")
        means.append(np.mean(results))
    print(f"all: {np.mean(means):>8.4f} {np.std(means):.4f}")
    print()

print("---\nGEF:")
for method in ["GCAM", "RAND"]:
    print(f"{method}")
    means = []
    for seed in ["912_model_1", "912_model_2", "912_model_3", "new_weights", "912"]:
        path_name = f"mutag/results_{seed}/faithfulness/{method}_GEF_node.npy"
        results = np.load(path_name)
        print(f"{seed:>15}: {np.nanmean(results):>8.4f} {np.nanstd(results):.4f}")
        means.append(np.nanmean(results))
    print(f"all: {np.mean(means):>8.4f} {np.std(means):.4f}")
    print()