"""
======================
SVM مع نواة مخصصة
======================

استخدام بسيط لآلة المتجهات الداعمة لتصنيف عينة. سيقوم
برسم سطح القرار وناقلات الدعم.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets, svm
from sklearn.inspection import DecisionBoundaryDisplay

# استيراد بعض البيانات للتجربة
iris = datasets.load_iris()
X = iris.data[:, :2]  # نأخذ فقط الخاصيتين الأوليين. يمكننا
# تجنب هذا التقطيع غير الجذاب باستخدام مجموعة بيانات ثنائية الأبعاد
Y = iris.target


def my_kernel(X, Y):
    """
    ننشئ نواة مخصصة:

                 (2  0)
    k(X, Y) = X  (    ) Y.T
                 (0  1)
    """
    M = np.array([[2, 0], [0, 1.0]])
    return np.dot(np.dot(X, M), Y.T)


h = 0.02  # حجم الخطوة في الشبكة

# ننشئ مثالاً لآلة المتجهات الداعمة ونقوم بضبط البيانات.
clf = svm.SVC(kernel=my_kernel)
clf.fit(X, Y)

ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    cmap=plt.cm.Paired,
    ax=ax,
    response_method="predict",
    plot_method="pcolormesh",
    shading="auto",
)

# رسم نقاط التدريب أيضاً
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolors="k")
plt.title("تصنيف من 3 فئات باستخدام آلة المتجهات الداعمة مع نواة مخصصة")
plt.axis("tight")
plt.show()