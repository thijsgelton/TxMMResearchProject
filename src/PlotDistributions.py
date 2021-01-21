import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    results = pd.read_csv("../data preprocessing/results.csv")
    for index, row in results.iterrows():
        distr = np.random.normal(row['overall_mean'], row['overall_std'], size=100000)
        sns.displot(distr, kind="kde")
        plt.show()
