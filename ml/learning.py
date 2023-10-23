import numpy as np
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

RANDOM_SEED = 0

def read_features_file(features_file):
    df_features = pd.read_csv(features_file, index_col='dr7objid')
    return df_features


def get_feat_target(df_features):

    X = df_features[['F{}'.format(i) for i in range(4096)]]
    y = df_features['target']

    return X, y


def split_features(df_features, ratio=0.25, random_state=RANDOM_SEED):

    X, y = get_feat_target(df_features)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=random_state)

    return X_train, X_test, y_train, y_test


def train_test_MLP(df_features, MLP_shape):

    X_train, X_test, y_train, y_test = split_features(df_features)
    
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                     hidden_layer_sizes=MLP_shape, random_state=1)

    clf.fit(X_train, y_train)
    
    y_test_predicted = clf.predict(X_test)    

    plot_contingency_table(y_test, y_test_predicted)

    return clf

def plot_hidden_layer(trained_MLP, n_layer, fig_shape, image_shape):

    cf = trained_MLP.coefs_[n_layer]
    
    plot_mosaic(np.transpose(cf), fig_shape, image_shape)


def plot_mosaic(table_arr, fig_shape, image_shape):

    # table_arr has the images to plot in rows
    if table_arr.shape[0] != fig_shape[0]*fig_shape[1]:
        raise Exception("dimensions do not match {} != {}*{}".format(table_arr.shape[0], fig_shape[0], fig_shape[1]))
    if table_arr.shape[1] != image_shape[0]*image_shape[1]:
        raise Exception("dimensions do not match {} != {}*{}".format(table_arr.shape[1], image_shape[0], image_shape[1]))

    fig = plt.figure()
    for i in range(fig_shape[0]*fig_shape[1]):
        arr = table_arr[i, :].reshape(image_shape)
        ax = fig.add_subplot(fig_shape[0], fig_shape[1], i + 1)
        ax.imshow(arr, cmap='gist_heat')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)

    plt.show()


def compute_PCA(X_train, n_components, plot=False):

    pca = PCA(n_components=n_components)

    pca.fit(X_train)

    if plot:
        plot_mosaic(pca.components_[:64, :], [8,8], [64, 64])

        plt.semilogy(np.cumsum(pca.explained_variance_ratio_))
        plt.grid()
        plt.axhline(y=0.9, color='r')
        plt.axhline(y=0.95, color='r')
        plt.axhline(y=0.99, color='r')
        plt.show()


    return pca
    

def train_test_logistic(df_features):

    X_train, X_test, y_train, y_test = split_features(df_features)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)

    y_test_predicted = clf.predict(X_test)    

    plot_contingency_table(y_test, y_test_predicted)

    return clf


def train_test_PCA_logistic(df_features, n_PCA_components):

    X_train, X_test, y_train, y_test = split_features(df_features)

    pca = compute_PCA(X_train, n_PCA_components)

    new_X_train = pca.transform(X_train)

    l = 0.3
    clf = LogisticRegression(
        fit_intercept=True,
        penalty='l2',
        C=1/l,
        max_iter=100,
        tol=1e-11,
        solver='lbfgs',
        verbose=1)
    clf.fit(new_X_train, y_train)

    new_X_test = pca.transform(X_test)
    y_test_predicted = clf.predict(new_X_test)

    plot_contingency_table(y_test, y_test_predicted)

    return clf



def plot_contingency_table(y_test, y_predict):

    TP = ((y_predict == 1) & (y_test == 1)).sum()
    FP = ((y_predict == 1) & (y_test != 1)).sum()
    TN = ((y_predict != 1) & (y_test != 1)).sum()
    FN = ((y_predict != 1) & (y_test == 1)).sum()
    
    total = len(y_test)

    print('''
          |           Sample         |
          |       P|       N|   Total|
    |    P|{: 8}|{: 8}|{: 8}|
Pred|    N|{: 8}|{: 8}|{: 8}|
    |Total|{: 8}|{: 8}|{: 8}|'''.format(TP, FP, TP + FP, FN, TN, TN + FN, TP + FN, FP + TN, TP + FP + FN + TN))

    print('accuracy={}\nprecission={}\nrecall={}'.format(float(TP + TN)/float(total),
                                                        TP/(TP + FP), TP/(TP + FN)))

