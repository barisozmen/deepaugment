import pandas as pd


def df_correlations(df, features):
    """ Calculates pearson and spearman correlations of each tuple of given features
        
    Args:
        df (pandas.DataFrame)
        features (list): features to calculate their correlations
    """
    rowList = []
    for i, feature_val in enumerate(features):
        for j in range(i + 1, len(features)):
            feat1 = feature_val
            feat2 = feature_val
            tmp_df = df[(df[feat1].notnull()) & (df[feat2].notnull())]
            p_r, p_p = pearsonr(tmp_df[feat1], tmp_df[feat2])
            s_r, s_p = spearmanr(tmp_df[feat1], tmp_df[feat2])
            rowList.append([feat1, feat2, p_r, p_p, s_r, s_p])

    return pd.DataFrame(
        rowList,
        columns=[
            "column1",
            "column2",
            "pearson_r",
            "pearson_pvalue",
            "spearman_r",
            "spearman_pvalue",
        ],
    )


def summary(df):
    """Summarizes the given dataframe
        
    Args:
        df (pandas.DataFrame): dataframe to be summarized
    """

    rowList = []
    for col in df.columns:
        rowList.append(
            [
                col,
                "{:.0f}%".format(df[col].notnull().sum() / len(df) * 100),
                df[col].nunique(),
                "{:.0f}%".format(df[col].nunique() / df[col].notnull().sum() * 100),
                df[col].dtypes,
                "N/A"
                if df[col].dtypes not in ["float", "int"]
                else "{:.2f}".format(df[col].mean()),
                "N/A"
                if df[col].dtypes not in ["float", "int"]
                else "{:.2f}".format(df[col].std()),
            ]
        )
    stats = pd.DataFrame(
        rowList,
        columns=[
            "column",
            "filled",
            "n_unique",
            "uniques/filled",
            "dtype",
            "mean",
            "std",
        ],
    )

    return stats


def rank_in_group(df, group_col, rank_col, rank_method="first"):
    """Ranks a column in each group which is grouped by another column
        
    Args:
        df (pandas.DataFrame): dataframe to rank-in-group its column
        group_col (str): column to be grouped by
        rank_col (str): column to be ranked for
        rank_method (str): rank method to be the "method" argument of pandas.rank() function
        
    Returns:
        pandas.DataFrame: dataframe after the rank-in-group operation
    """

    df = df.copy()
    df_slice = df[[group_col, rank_col]].drop_duplicates()
    df_slice["ranked_{}".format(rank_col)] = df_slice[rank_column].rank(
        method=rank_method
    )
    df = pd.merge(
        df,
        df_slice[[group_col, "ranked_{}".format(rank_col)]],
        how="left",
        on=group_col,
    )
    return df
