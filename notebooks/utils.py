import cv2
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
import numpy as np
import pandas as pd
import pickle
import seaborn as sns
from collections import defaultdict


def parse_log(file):
    nframes = {
        'fish': 600,
        'mouse': 2330,
        'marmosets': 15000,
        'parenting': 2671,
    }

    temp = defaultdict(lambda: defaultdict(list))
    with open(file) as f:
        for line in f.readlines():
            if not line.startswith('DEBUG'):
                continue
            dataset, size, _, t = line.strip()[11:].split('-')
            temp[dataset][int(size)].append(nframes[dataset] / float(t))
    results = defaultdict(dict)
    for dataset, dict_ in temp.items():
        for size, l in dict_.items():
            results[dataset][size] = sum(l) / len(l)
    return results, temp


def calc_detection_performance(
    results_file,
    metadata_file,
    pcutoff,
    tol,
    inds=None,
):
    with open(metadata_file, "rb") as file:
        dict_ = pickle.load(file)
        test_inds = dict_["data"]["testIndices"]
        bpts = dict_["data"]["DLC-model-config file"]["all_joints_names"]
    temp = (
        pd.read_csv(results_file, header=[0, 1, 2, 3], index_col=0)
        .droplevel("scorer", axis=1)
        .iloc[:, :-1]
    )
    if inds is not None:
        temp = temp.iloc[inds]
    df = (
        temp.stack(["individuals", "bodyparts"])
        .reset_index()
        .rename(columns={"level_0": "inds"})
    )
    df = df[~df["bodyparts"].isin(["dfin1", "dfin2"])]
    df.loc[df["conf"] < pcutoff, "rmse"] = np.nan
    df["train"] = "train"
    df.loc[df["inds"].isin(test_inds), "train"] = "test"
    x = np.sort(df["rmse"].dropna())
    y = np.linspace(0, 1, x.size)
    th = x[np.argmax(y >= tol)]
    df.loc[df["rmse"] >= th, "rmse"] = np.nan
    df["bodyparts"] = df["bodyparts"].astype("category")
    bpts = [bpt for bpt in bpts if bpt in df["bodyparts"].cat.categories]
    df["bodyparts"] = df["bodyparts"].cat.reorder_categories(bpts)
    df["bodyparts"] = df["bodyparts"].apply(str.lower)
    return df.sort_values(by="bodyparts")


def plot_single_image_keypoints(
    datafile,
    imgfile,
    markersize=8,
    colormap="viridis",
    output_name="",
):
    data = pd.read_hdf(datafile)
    frame = cv2.imread(imgfile)
    frame = frame[..., ::-1] / 255
    fig, ax = plt.subplots()
    ax.imshow(frame)
    coords = []
    for n, (_, df_) in enumerate(data.groupby("individuals")):
        temp = df_.to_numpy().reshape((-1, 3))
        coords.append(np.c_[temp, np.ones(temp.shape[0]) * n])
    coords = np.concatenate(coords)
    coords = coords[~np.isnan(coords).any(axis=1)]
    n_animals = len(np.unique(coords[:, -1]))
    cmap = plt.cm.get_cmap(colormap, n_animals)
    colors = cmap(coords[:, -1].astype(int))
    ax.scatter(*coords[:, :2].T, c=colors, s=markersize)
    ax.axis("off")
    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    plt.savefig(output_name or "image.png", bbox_inches=extent, dpi=600)


def plot_detection_performance(
    results_file,
    metadata_file,
    pcutoff,
    tol=0.995,
    pck_threshold=None,
    cmap="Blues",
    plot_type='boxplot',
    ax=None,
    inds=None,
):
    df = calc_detection_performance(
        results_file,
        metadata_file,
        pcutoff,
        tol,
        inds,
    )

    if isinstance(pck_threshold, pd.Series):
        df["pck"] = pck_threshold[
            list(map(tuple, df[["individuals", "inds"]].values))
        ].to_numpy()
    else:
        df["pck"] = pck_threshold

    def _calc_pck(df):
        rmse = df[["rmse", "pck"]].dropna().to_numpy()
        return np.sum(rmse[:, 0] <= rmse[:, 1]) / rmse.shape[0]

    pck = df.groupby(["bodyparts", "train"]).apply(_calc_pck)

    if ax is None:
        _, ax = plt.subplots(tight_layout=True, figsize=(3, 7))
    with sns.color_palette(cmap, 2):
        if plot_type == 'box':
            sns.boxplot(
                y="bodyparts",
                x="rmse",
                hue="train",
                hue_order=["train", "test"],
                showfliers=False,
                data=df,
                linewidth=1,
                ax=ax,
                orient="h",
            )
        else:
            sns.violinplot(
                y="bodyparts",
                x="rmse",
                hue="train",
                hue_order=["train", "test"],
                showfliers=False,
                split=True,
                cut=0,
                inner="quartile",
                data=df,
                scale="width",
                linewidth=1,
                ax=ax,
                orient="h",
            )
    if pck_threshold is not None:
        trans = transforms.blended_transform_factory(ax.transAxes, ax.transData)
        for i, (_, values) in enumerate(pck.groupby("bodyparts")):
            try:
                ax.text(
                    1,
                    i - 0.4,
                    f'{values.loc[:, "train"].item():.2f}',
                    c="gray",
                    size="small",
                    va="top",
                    ha="right",
                    transform=trans,
                )
            except KeyError:
                pass
            try:
                ax.text(
                    1,
                    i + 0.4,
                    f'{values.loc[:, "test"].item():.2f}',
                    c="gray",
                    size="small",
                    va="bottom",
                    ha="right",
                    transform=trans,
                )
            except KeyError:
                pass
    ax.set_ylabel("")
    ax.set_xlabel("RMSE (pixels)")
    ax.tick_params(axis="y", length=0)
    for side, spine in ax.spines.items():
        if side != "bottom":
            spine.set_visible(False)
    ax.get_legend().remove()
    return df, pck
