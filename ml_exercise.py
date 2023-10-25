# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# %matplotlib inline

import numpy as np
import matplotlib.pyplot as plt

from ml import plots
from ml import learning
# -

# # Exercise
#
# If you don't have the `T_F_DR14_ZooSpec_10000.csv` file in the `resources` folder, uncomment an run the cells below to download and extract it. It is big (95MB), so it can take some time.
#
# If running the cells doesn't work, just navigate to the link in your internet browser, download and decompress the file, and move it to the resources folder.

# +
# #!wget https://public.pic.es/s/6p3loHQQvOXPm00/download -O resources/T_F_DR14_ZooSpec_10000.zip

# +
# #!unzip resources/T_F_DR14_ZooSpec_10000.zip -d resources
# -

df_features = learning.read_features_file('resources/T_F_DR14_ZooSpec_10000.csv')

df_features.head()

# What is this?

X, y = learning.get_feat_target(df_features)

learning.plot_mosaic(X.values[:16, :], [4,4], [64, 64])

# This dataset contains pictures of galaxies and their shape class. It was extracted from the [Galaxy Zoo Project](https://www.zooniverse.org/projects/zookeeper/galaxy-zoo/)
#
# * dr7objid: Id of the object in Galaxy Zoo
# * target: class of galaxy
#   * 0: undefined
#   * 1: elliptical
#   * 2: spiral
# * F0 to F4095: 64x64 galaxy image arranged into an array of 4096 normalized values
#
# **Build a ML algorithm to classify galaxies into elliptical or spiral based on this dataset with the best performance possible**
#
# Some clues:
# * Do we need to do some preprocessing? Filtering? Feature Scaling?
# * What's dimensionality of the data? size of the sample vs features
# * We didn't talk about it, but scikit-learn includes neural networks also: https://scikit-learn.org/stable/modules/neural_networks_supervised.html

# # Possible solutions

# ## Savage SVM
#
# Run Support Vector Machine on the raw data.
#
# We don't filter out the undefined galaxies and we don't do dimensionality reduction. SVM will increase the number of dimensions anyway so ...
#
# The features are already scaled.

# +
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
# -

from sklearn.svm import SVC
clf = SVC(kernel='rbf')
clf.fit(X_train, y_train)

# +
y_prediction = clf.predict(X_test)

plots.plot_bars_and_confusion(truth=y_test, prediction=y_prediction);
# -

# We a quite good accuracy, but if we restrict to elliptical/spiral, the performance is horrible. The labels in the confusion matrix are not correct because we have more than two classes.

# +
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_prediction)

a = np.diag(cm)[1:].sum()/cm[1:, :].sum()
print(f'Accuracy on Spirals/ellipticals = {a:.0%}')
# -

# ## Savage Random Forests
#
# Random forests are not so affected by high dimensionality, so we can give it a try. However, the underlying decision trees will look at each pixel individually and split the dataset depending on their values

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=20, max_depth=100)
rf.fit(X_train, y_train)

y_prediction = rf.predict(X_test)
plots.plot_bars_and_confusion(truth=y_test, prediction=y_prediction)

# Again if we restrict to the spiral/elliptical galaxies, the accuracy is horrible

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_prediction)

a = np.diag(cm)[1:].sum()/cm[1:, :].sum()
print(f'Accuracy on Spirals/ellipticals = {a:.0%}')

# ## Filtered SVM
#
# We can consider that the unlabelled galaxies are actually noise, so we don't want to take them into account. 
#
# Let's filter them out and try again

defined_mask = df_features.target != 0
filtered_df = df_features[defined_mask]

Xf, yf = learning.get_feat_target(filtered_df)

Xf_train, Xf_test, yf_train, yf_test = train_test_split(Xf, yf, test_size=0.3)

clf = SVC(kernel='rbf')
clf.fit(Xf_train, yf_train)

# +
yf_prediction = clf.predict(Xf_test)

plots.plot_bars_and_confusion(truth=yf_test, prediction=yf_prediction);
# -

# ### !!BOOOOM!!

# ## Filtered Random Forests
#
# You can play with the number of estimators and the maximum depth of the trees

rf = RandomForestClassifier(n_estimators=20, max_depth=100)
rf.fit(Xf_train, yf_train)

# +
yf_prediction = rf.predict(Xf_test)

plots.plot_bars_and_confusion(truth=yf_test, prediction=yf_prediction);
# -

# ### !!BOOM!!

# ## Dimensionality reduction + SVM
#
# The number of features wrt the number of samples is very high, 4096 vs 3701. So there's the possibility that we could have some benefit from applying dimensionality reduction (although the performance is good already).

from ml.learning import compute_PCA

dr = compute_PCA(Xf, n_components=256, plot=True)

Xr_train = dr.transform(Xf_train)

clf = SVC(kernel='rbf', probability=True)
clf.fit(Xr_train, yf_train)

# +
Xr_test = dr.transform(Xf_test)
yr_prediction = clf.predict(Xr_test)

plots.plot_bars_and_confusion(truth=yf_test, prediction=yr_prediction);
# -

# ### BOOOOOOOOOM !!!!!

# ## Dimensionality reduction + Random Forests

rf = RandomForestClassifier(n_estimators=20, max_depth=100)
rf.fit(Xr_train, yf_train)

# +
yr_prediction = rf.predict(Xr_test)

plots.plot_bars_and_confusion(truth=yf_test, prediction=yr_prediction);


# -

# Performance drops ... why? loss of information?

# ## Apply the model to the unlabeled data

def plot_mosaic_vs_prediction(table_arr, pred, proba, fig_shape, image_shape):

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
        ax.text(0, image_shape[1], f'{pred[i]}: {proba[i, pred[i] - 1]:.0%}', color='white')
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)



unlabeled_df = df_features[~defined_mask]

Xu, yu = learning.get_feat_target(unlabeled_df)

Xur = dr.transform(Xu)

yur = clf.predict(Xur)
yur_proba = clf.predict_proba(Xur)

plot_mosaic_vs_prediction(Xu.values[:16, :], yur, yur_proba, [4,4], [64, 64])
print('1 == elliptical\n2 == spiral')


