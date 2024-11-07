"""
=====================================
رسم المتجهات الداعمة في LinearSVC
=====================================

على عكس SVC (الذي يعتمد على LIBSVM)، فإن LinearSVC (الذي يعتمد على LIBLINEAR) لا يوفر
المتجهات الداعمة. يوضح هذا المثال كيفية الحصول على المتجهات الداعمة في LinearSVC.

"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.svm import LinearSVC

X, y = make_blobs(n_samples=40, centers=2, random_state=0)

plt.figure(figsize=(10, 5))
for i, C in enumerate([1, 100]):
    # "hinge" هي خسارة SVM القياسية
    clf = LinearSVC(C=C, loss="hinge", random_state=42).fit(X, y)
    # الحصول على المتجهات الداعمة من خلال دالة القرار
    decision_function = clf.decision_function(X)
    # يمكننا أيضًا حساب دالة القرار يدويًا
    # decision_function = np.dot(X, clf.coef_[0]) + clf.intercept_[0]
    # المتجهات الداعمة هي العينات التي تقع داخل حدود الهامش
    # والتي يُحافظ على حجمها تقليديًا عند 1
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = X[support_vector_indices]

    plt.subplot(1, 2, i + 1)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        ax=ax,
        grid_resolution=50,
        plot_method="contour",
        colors="k",
        levels=[-1, 0, 1],
        alpha=0.5,
        linestyles=["--", "-", "--"],
    )
    plt.scatter(
        support_vectors[:, 0],
        support_vectors[:, 1],
        s=100,
        linewidth=1,
        facecolors="none",
        edgecolors="k",
    )
    plt.title("C=" + str(C))
plt.tight_layout()
plt.show()