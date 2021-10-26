
def pd_custom_sort_values(df, sorter, column):
    """
    Sorts a pandas dataframe in the order of the elements
    in the list in sorter.
    :param df: dataframe
    :param sorter: list of elements (must contain all)
    :param column: column to be used
    :return:
    """
    sort_d={}
    ii=0
    for s in sorter:
        sort_d[s]=ii
        ii+=1
    sort_ind='sort_ind'
    df[sort_ind]= df[column].apply(lambda x: sort_d[x] )
    df = df.sort_values('sort_ind')
    return df.drop(sort_ind, axis=1)
