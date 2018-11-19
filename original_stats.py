# coding: utf-8
#%matplotlib inline

# 相関比の算出
def corr_ratio(df, col_c, col_n):
    import pandas as pd
    import math
    
    # 抽出
    df = df[[col_c, col_n]]
    
    #-------------
    # 級内変動 Sw
    #-------------
    
    # 集計
    agg = df.groupby(col_c)

    # 級別平均
    agg_by_class = agg.sum()
    count_by_class = agg.count()
    means_by_class = agg_by_class / count_by_class

    # 各レコードのクラス平均に対する偏差平方和の合計
    sw = df.apply(lambda row: math.pow(row[col_n] - means_by_class.T[row[col_c]], 2), axis=1).sum()

    #-------------
    # 級間変動 Sb
    #-------------

    # 全体平均
    mean = df[col_n].mean()

    # 各クラス平均の全体平均に対する偏差の重み付き平方和
    tmp = (means_by_class - mean).apply(lambda row: pd.Series(math.pow(row, 2)), axis=1)
    tmp.columns = [col_n]
    sb = (tmp * count_by_class).sum()
    
    #-------------
    # 相関比 Eta^2
    #-------------    
    
    eta2 = sb / (sw + sb)
    
    return eta2[0]


# クラメールの連関係数の算出
def cramers_v(df):
    from scipy.stats import chi2_contingency
    import math
    
    # カイ二乗統計量
    chi2, p_value, dof = chi2_contingency(df)[0:3]
    
    # サンプル数
    n = df.sum().sum()
    
    # クラメールの連関係数
    v = math.sqrt(chi2 / n / min(df.shape[0]-1, df.shape[1]-1))
    
    return v, chi2, p_value, dof

# クロス集計してからクラメールの連関係数の算出
def crossagg_and_cramers_v(df, colname_x, colname_y):
    import pandas as pd
    
    # 抽出
    df = df[[colname_x, colname_y]]
    
    # クロス集計
    agg = pd.crosstab(index=df[colname_y], columns=df[colname_x])
    
    return cramers_v(agg)


# 独立変数をダミー変数にして従属変数とクロス集計してからクラメールの連関係数の算出
def getdummy_crossagg_cramers_v(df, colname_x, col_y):
    
    onehot = pd.get_dummies(df[colname_x])
    colnames = onehot.columns

    res = ()
    for i in colnames:
        tmp = (i,)
        tmp += cramers_v(pd.crosstab(col_y, onehot[i]))
        res += (tmp,)
            
    return res

# クロス集計してヒートマップ
def crossagg_and_heatmap(df, colname_x, colname_y, file):
    import seaborn as sns
    import matplotlib.pyplot as plt

    tmp = pd.crosstab(index=df[colname_y], columns=df[colname_x])
    plt.figure(figsize=(len(tmp.index)*2, 5))
    sns.heatmap(data=tmp,\
                annot=True, annot_kws={"fontsize":"large"},\
                cmap="viridis", linecolor="white", linewidth=0.01)
    
    if file == True:
        plt.savefig("./" + colname_y + "_" + colname_x + "_heatmap.png")
        
    plt.show()


# バイオリンプロット
def violin(df, colname_x, col_y, file):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    onehot = pd.get_dummies(df[colname_x])
    colnames = onehot.columns
    
    sns.violinplot(data=pd.concat([col_y, df[colname_x]], axis=1), x=colname_x, y=col_y.name)
    
    if file == True:
        plt.savefig("./" + col_y
                    .name + "_" + colname_x + "_violin.png")
    
    plt.show()



