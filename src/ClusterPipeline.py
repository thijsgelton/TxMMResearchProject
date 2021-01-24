import gower
import pandas as pd
from sklearn import metrics
from sklearn.metrics.cluster import rand_score
from sklearn.cluster import SpectralClustering

ARTIST_GT = 2

NUMBER_OF_PRICE_BINS = 3

encodings = [f'encoded_{i}' for i in range(30)]

features = ['width_painting_cm', 'kunstenaar', 'height_painting_cm', 'width_frame_cm', 'height_frame_cm', 'condition',
            'technique', 'signed', 'period', 'style', 'subject', 'price', 'price_binned']

feature_groups = {
    "size": ['width_painting_cm', 'height_painting_cm', 'width_frame_cm', 'height_frame_cm'],
    "condition": ["condition"],
    "technique": ["technique"],
    "time": ["period"],
    "signed": ["signed"],
    "artist": ["kunstenaar"],
    "style_description": ["style"],
    "style_patterns": encodings,
    "style_combined": [*encodings, "style"],
    "subject": ["subject"],
    "authenticity": ["signed", "kunstenaar"],
    "all_features": [*features, *encodings],
    "metadata": features
}


def filter_artist_gt(gt=1):
    global data
    col_name = f"artist_gt_{gt}"
    artists = (data['kunstenaar'].value_counts() > gt).rename(col_name)
    data = data.merge(artists.to_frame(), left_on="kunstenaar", right_index=True)
    data = data[data[col_name] == True]


if __name__ == '__main__':
    clustering_metrics = [
        metrics.homogeneity_score,
        metrics.completeness_score,
        metrics.v_measure_score,
        metrics.adjusted_rand_score,
        metrics.adjusted_mutual_info_score,
    ]

    for artist_gt in range(ARTIST_GT):
        data = pd.read_csv("../data preprocessing/final_df_with_encodings_with_price_binned.csv", header=0,
                           usecols=[*features, *encodings])
        filter_artist_gt(artist_gt)
        n_rows = len(data)
        mapping = {"(0, 250]": 0, "(250, 1250]": 1, "(1250, 4200]": 2}
        labels_true = [mapping[x] for x in data['price_binned']]
        results = []
        for name, features_ in feature_groups.items():
            result = [name]
            features_used = gower.gower_matrix(data[features_])
            cluster = SpectralClustering(NUMBER_OF_PRICE_BINS, affinity='precomputed', n_init=200, n_jobs=-1).fit(
                features_used)
            labels_pred = cluster.labels_
            result += [m(labels_true, labels_pred) for m in clustering_metrics]
            results.append(result)
        print(pd.DataFrame(results,
                           columns=["Feature",
                                    "V-measure",
                                    "Adj. Rand Index"]).sort_values(by='Adj. Rand Index', ascending=False).to_latex(
            f"../data preprocessing/artist_gt_{artist_gt}_ari.tex", index=False, caption=f"Number of rows {n_rows}"))
