def semi_join(df1, df2, on):
    """
    Semi-join. Returns all elements in df1 that match in df2 by on
    """
    # https://stackoverflow.com/questions/63660610/how-to-perform-semi-join-with-multiple-columns-in-pandas
    return df1[df1[on].agg(tuple, 1).isin(df2[on].agg(tuple, 1))].reset_index(drop=True)
