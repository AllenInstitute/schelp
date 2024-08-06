import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc


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
    df_ = df_.sort_values(
        by=["Overall AD neuropathological Change", "supertype_entropy"],
        ascending=[True, False],
    )

    # find the top donor_frac_pergroup of donors donors with the most entropy of supertypes within each in each value of 'Overall AD neuropathological Change'
    df_test = (
        df_.groupby("Overall AD neuropathological Change")[df_.columns]
        .apply(
            lambda x: x.nlargest(int(len(x) * donor_frac_pergroup), "supertype_entropy"),
            include_groups=True,
        )
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


def make_donor_splits_dataset(paths, donor_frac_pergroup=0.15):
    """Creates a train set from the balanced data, and a test set from the full singleome
    dataset using a subset of donors. The donors in the test set are removed from the training set.
    """
    print("loading singleome data")
    adata_singleome = sc.read(
        paths["frozen"] / "SEAAD_MTG_RNAseq_Singleome_final-nuclei.2024-02-13.h5ad",
        backed="r",
    )
    train_ind, test_ind = donor_split(obs=adata_singleome.obs, donor_frac_pergroup=donor_frac_pergroup)

    print("loading balanced data")
    adata_balanced = sc.read_h5ad(
        str(paths["data"]) + "/Human-Brain/balanced_SEAAD_MTG_RNAseq_Singleome_final-nuclei.2024-06-18.h5ad"
    )
    # removing donors that are in the test set
    test_donors = adata_singleome.obs["Donor ID"].loc[test_ind].unique()
    adata_balanced = adata_balanced[~adata_balanced.obs["Donor ID"].isin(test_donors)]

    print("calculating variable genes")
    # calculate variable genes based on training data
    sc.pp.highly_variable_genes(adata_balanced, n_top_genes=4000, flavor="cell_ranger", batch_key="Donor ID")
    adata_balanced = adata_balanced[:, adata_balanced.var["highly_variable"]]

    # save adata_balanced to disk
    adata_balanced.write(str(paths["data"]) + "/Human-Brain/balanced_donor_split_train_v1.h5ad")
    print(f"Train data shape: {adata_balanced.shape}")
    features = adata_balanced.var.index
    del adata_balanced

    print("loading subsetted singleome data to memory")
    # get the test data into memory
    adata_test = adata_singleome[test_ind, features]
    adata_test = adata_test.to_memory()
    adata_test.write(str(paths["data"]) + "/Human-Brain/donor_split_test_v1.h5ad")
    print(f"Test data shape: {adata_test.shape}")

    return adata_balanced, adata_test


def add_scgpt_data_columns(adata):
    """Includes celltype, celltype_id, batch, batch_id, gene_name columns in the adata object.
    These are used in the scgpt scripts, so we create them here as a convenience.
    """
    adata.obs["celltype"] = adata.obs["Supertype"]
    adata.obs["batch"] = adata.obs["Donor ID"]
    num_types = adata.obs["celltype"].unique().size
    id2type = dict(enumerate(adata.obs["celltype"].cat.categories))
    celltypes = adata.obs["celltype"].unique()
    celltype_id_labels = adata.obs["celltype"].astype("category").cat.codes.values
    adata.obs["celltype_id"] = celltype_id_labels
    adata.obs["batch_id"] = adata.obs["batch"].cat.codes.values
    adata.var["gene_name"] = adata.var.index.tolist()
    return adata, num_types, id2type, celltypes
