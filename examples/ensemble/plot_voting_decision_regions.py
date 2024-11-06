"""
==================================================
رسم حدود القرار لـ VotingClassifier
==================================================

.. currentmodule:: sklearn

رسم حدود القرار لـ :class:`~ensemble.VotingClassifier` لميزتين من مجموعة بيانات Iris.

رسم احتمالات الفئة للعينة الأولى في مجموعة بيانات تجريبية تم التنبؤ بها بواسطة
ثلاثة مصنفات مختلفة وتم حساب متوسطها بواسطة
:class:`~ensemble.VotingClassifier`.

أولاً، يتم تهيئة ثلاثة مصنفات نموذجية
(:class:`~tree.DecisionTreeClassifier`،
:class:`~neighbors.KNeighborsClassifier`، و :class:`~svm.SVC`) وتستخدم
لتهيئة :class:`~ensemble.VotingClassifier` للتصويت الناعم مع أوزان `[2،
1، 2]`، مما يعني أن الاحتمالات المتوقعة لـ
:class:`~tree.DecisionTreeClassifier` و :class:`~svm.SVC` يتم احتساب كل منها مرتين
بقدر أوزان مصنف :class:`~neighbors.KNeighborsClassifier`
عندما يتم حساب الاحتمال المتوسط.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

from itertools import product

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.ensemble import VotingClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# تحميل بعض بيانات المثال
iris = datasets.load_iris()
X = iris.data[:, [0, 2]]
y = iris.target

# تدريب المصنفات
clf1 = DecisionTreeClassifier(max_depth=4)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma=0.1, kernel="rbf", probability=True)
eclf = VotingClassifier(
    estimators=[("dt", clf1), ("knn", clf2), ("svc", clf3)],
    voting="soft",
    weights=[2, 1, 2],
)

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# رسم مناطق القرار
f, axarr = plt.subplots(2, 2, sharex="col", sharey="row", figsize=(10, 8))
for idx, clf, tt in zip(
    product([0, 1], [0, 1]),
    [clf1, clf2, clf3, eclf],
    ["شجرة القرار (العمق = 4)", "KNN (k = 7)", "Kernel SVM", "التصويت الناعم"],
):
    DecisionBoundaryDisplay.from_estimator(
        clf, X, alpha=0.4, ax=axarr[idx[0], idx[1]], response_method="predict"
    )
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()


