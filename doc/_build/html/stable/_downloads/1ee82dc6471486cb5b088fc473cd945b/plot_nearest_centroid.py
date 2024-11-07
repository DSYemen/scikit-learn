"""
===============================
تصنيف أقرب مركز
===============================

استخدام عينة لتصنيف أقرب مركز.
سيقوم برسم حدود القرار لكل فئة.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap

from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.neighbors import NearestCentroid

# استيراد بعض البيانات للتجربة
iris = datasets.load_iris()
# نأخذ فقط أول ميزتين. يمكننا تجنب هذا التقطيع غير المناسب
# باستخدام مجموعة بيانات ثنائية الأبعاد
X = iris.data[:, :2]
y = iris.target

# إنشاء خرائط الألوان
cmap_light = ListedColormap(["orange", "cyan", "cornflowerblue"])
cmap_bold = ListedColormap(["darkorange", "c", "darkblue"])

for shrinkage in [None, 0.2]:
    # ننشئ مثالاً لتصنيف أقرب مركز ونقوم بضبط البيانات.
    clf = NearestCentroid(shrink_threshold=shrinkage)
    clf.fit(X, y)
    y_pred = clf.predict(X)
    print(shrinkage, np.mean(y == y_pred))

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf, X, cmap=cmap_light, ax=ax, response_method="predict"
    )

    # رسم نقاط التدريب أيضًا
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor="k", s=20)
    plt.title("تصنيف من 3 فئات (shrink_threshold=%r)" % shrinkage)
    plt.axis("tight")

plt.show()