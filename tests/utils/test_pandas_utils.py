import pandas as pd

from elexmodel.utils import pandas_utils


def test_semi_join():
    df1 = pd.DataFrame({"c1": ["a", "b", "c"], "c2": [1, 2, 3], "c3": [7, 8, 9]})

    df2 = pd.DataFrame({"c1": ["b", "d"], "c2": [5, 6]})
    df3 = pandas_utils.semi_join(df1, df2, ["c1"])
    assert pd.DataFrame({"c1": ["b"], "c2": [2], "c3": [8]}).equals(df3)

    df2 = pd.DataFrame({"c1": ["x", "d"], "c2": [5, 6]})
    df3 = pandas_utils.semi_join(df1, df2, ["c1"])
    assert df3.empty

    df2 = pd.DataFrame({"c1": ["b", "c"], "c2": [5, 6]})
    df3 = pandas_utils.semi_join(df1, df2, ["c1"])
    assert pd.DataFrame({"c1": ["b", "c"], "c2": [2, 3], "c3": [8, 9]}).equals(df3)

    df2 = pd.DataFrame({"c1": ["b", "a"], "c2": [5, 6]})
    df3 = pandas_utils.semi_join(df1, df2, ["c1"])
    assert pd.DataFrame({"c1": ["a", "b"], "c2": [1, 2], "c3": [7, 8]}).equals(df3)
