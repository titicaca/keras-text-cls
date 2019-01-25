import logging


def flatten_list_dfs(x):
    res = []
    for i in x:
        if isinstance(i, list):
            res.extend(flatten_list_dfs(i))
        else:
            res.append(i)
    return res
