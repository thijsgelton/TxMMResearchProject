import gower
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.cluster import DBSCAN, SpectralClustering

encodings = [f'encoded_{i}' for i in range(50)]

features = ['width_painting_cm', 'height_painting_cm', 'width_frame_cm', 'height_frame_cm', 'condition', 'technique',
            'signed', 'period', 'style', 'subject', 'price', *encodings]

feature_groups = {
    "size": ['width_painting_cm', 'height_painting_cm', 'width_frame_cm', 'height_frame_cm'],
    "condition": ["condition"],
    "technique": ["technique"],
    "time": ["period"],
    "signed": ["signed"],
    "style_description": ["style"],
    "style_patterns": encodings,
    "subject": ["subject"]
}

if __name__ == '__main__':
    data = pd.read_csv("../data preprocessing/final_df_with_encodings.csv", header=0, usecols=features)

    X = data.drop('price', axis=1)
    y = data['price']
    results = []
    for name, features in feature_groups.items():
        ablated = gower.gower_matrix(X.drop(features, axis=1))
        cluster = SpectralClustering(8, affinity='precomputed').fit(ablated)
        clusters = cluster.labels_
        means = []
        stds = []
        unique_clusters = np.unique(clusters)
        for label in unique_clusters:
            prices = y[clusters == label].values
            mean = np.mean(prices)
            means.append(mean)
            std = np.std(prices)
            stds.append(std)
            print(f"Label {label} has a: \nMean: {mean}\nStandard Deviation: {std}\n")
        print("-------------------------------------------------------------------------------------")
        overall_mean = np.array(means).sum()
        overall_std = np.array(stds).sum()
        print(f"Overall mean {overall_mean} and overall standard deviation {overall_std}")
        results.append({
            "ablated_group": name,
            "ablated_features": features,
            "number_of_clusters": len(unique_clusters),
            "overall_mean": overall_mean,
            "overall_std": overall_std
        })
    pd.DataFrame(results).to_csv("../data preprocessing/results.csv", index=False)
