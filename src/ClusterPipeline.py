import gower
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn import metrics
from sklearn.cluster import DBSCAN

features = ['width_painting_cm', 'height_painting_cm', 'width_frame_cm', 'height_frame_cm', 'condition', 'technique',
            'signed', 'framed', 'period', 'style', 'subject', 'price', *[f'encoded_{i}' for i in range(10)]]

if __name__ == '__main__':
    data = pd.read_csv("../data preprocessing/final_df_with_encodings.csv", header=0, usecols=features)

    X = gower.gower_matrix(data.drop('price', axis=1))
    y = data['price']

    cluster = DBSCAN(eps=.04, metric="precomputed", min_samples=3).fit(X)
    clusters = cluster.labels_
    means = []
    stds = []
    for label in np.unique(clusters):
        prices = y[clusters == label].values
        mean = np.mean(prices)
        means.append(mean)
        std = np.std(prices)
        stds.append(std)
        print(f"Label {label} has a: \nMean: {mean}\nStandard Deviation: {std}\n")
    print("-------------------------------------------------------------------------------------")
    print(f"Overall mean {np.mean(np.array(means))} and overall standard deviation {np.std(np.array(stds))}")

    # #############################################################################
    # Compute DBSCAN
    core_samples_mask = np.zeros_like(cluster.labels_, dtype=bool)
    core_samples_mask[cluster.core_sample_indices_] = True
    labels = cluster.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    print("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(X, labels))

    # #############################################################################
    # Plot result
    import matplotlib.pyplot as plt

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = X[class_member_mask & core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)

        xy = X[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.show()
