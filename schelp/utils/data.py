import numpy as np
import anndata as ad
import scanpy as sc
import pandas as pd


def donor_split(obs, donor_frac_pergroup=0.15):
    """
    Split the data into train and test sets based on the entropy of supertypes within each value of 'Overall AD neuropathological Change'.
    `donor_frac_pergroup` of donors with the most entropy of supertypes within each group of `Overall AD neuropathological Change` are selected for the test set.

    Args:
        obs : pd.DataFrame
            The obs dataframe of the AnnData object.
        donor_frac_pergroup : float, optional

    Returns:
        train_idx : pd.Index 
        test_idx : pd.Index
    """

    # make a copy for internal function use. 
    df = obs[["Overall AD neuropathological Change", "Donor ID", "Supertype"]].copy()

    def entropy(x, eps=1e-12):
        p = x.value_counts(normalize=True)
        p = p + eps
        return -np.sum(p * np.log2(p))

    df_supertype_entropy = df.groupby("Donor ID")["Supertype"].apply(entropy).sort_values(ascending=False).to_frame()
    df_supertype_entropy.columns = ["supertype_entropy"]
    df_supertype_entropy.reset_index(inplace=True)

    df_ = df[["Overall AD neuropathological Change", "Donor ID"]].drop_duplicates()
    df_.reset_index(drop=True, inplace=True)

    # Merge df_ and df_supertype_entropy on Donor ID
    df_ = df_.merge(df_supertype_entropy, on="Donor ID")

    # sort by "supertype_entropy" and "Overall AD neuropathological Change"
    df_ = df_.sort_values(by=["Overall AD neuropathological Change", "supertype_entropy"], ascending=[True, False])

    # find the top donor_frac_pergroup of donors donors with the most entropy of supertypes within each in each value of 'Overall AD neuropathological Change'
    df_test = (
        df_.groupby("Overall AD neuropathological Change")[df_.columns]
        .apply(lambda x: x.nlargest(int(len(x) * donor_frac_pergroup), "supertype_entropy"), include_groups=True)
        .reset_index(drop=True)
    )

    # get total number of cells per donor in df, and merge results with the df_test
    df_total_cells = df.groupby("Donor ID").size().to_frame().reset_index()
    df_total_cells.columns = ["Donor ID", "total_cells"]
    df_test = df_test.merge(df_total_cells, on="Donor ID")

    perc_cells = (df_test["total_cells"].sum() / df.shape[0]) * 100
    print(f"High supertype entropy donors make up {perc_cells:.2f}% of total cells")

    test_idx = df[df["Donor ID"].isin(df_test["Donor ID"])].index
    train_idx = df[~df["Donor ID"].isin(df_test["Donor ID"])].index

    test_frac = (test_idx.shape[0] / df.shape[0]) * 100
    train_frac = (train_idx.shape[0] / df.shape[0]) * 100
    print(f"Train set: {train_frac:.2f}%")
    print(f"Test set: {test_frac:.2f}%")

    return train_idx, test_idx




