"""
=========================================
SVM: المستوى الفاصل ذو الهامش الأقصى
=========================================

ارسم المستوى الفاصل ذو الهامش الأقصى ضمن مجموعة بيانات قابلة للفصل من فئتين باستخدام مصنف آلة المتجهات الداعمة مع نواة خطية.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.datasets import make_blobs
from sklearn.inspection import DecisionBoundaryDisplay

# نقوم بإنشاء 40 نقطة قابلة للفصل
X, y = make_blobs(n_samples=40, centers=2, random_state=6)

# تدريب النموذج، عدم استخدام التنظيم لأغراض التوضيح
clf = svm.SVC(kernel="linear", C=1000)
clf.fit(X, y)

plt.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=plt.cm.Paired)

# رسم دالة القرار
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    plot_method="contour",
    colors="k",
    levels=[-1, 0, 1],
    alpha=0.5,
    linestyles=["--", "-", "--"],
    ax=ax,
)
# رسم المتجهات الداعمة
ax.scatter(
    clf.support_vectors_[:, 0],
    clf.support_vectors_[:, 1],
    s=100,
    linewidth=1,
    facecolors="none",
    edgecolors="k",
)
plt.show()