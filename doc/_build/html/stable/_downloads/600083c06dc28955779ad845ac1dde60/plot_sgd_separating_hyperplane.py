"""
=========================================
SGD: المستوى الفاصل ذو الهامش الأقصى
=========================================

ارسم المستوى الفاصل ذو الهامش الأقصى ضمن مجموعة بيانات ثنائية الفصل
باستخدام مصنف آلات المتجهات الداعمة الخطي الذي تم تدريبه باستخدام SGD.

"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_blobs
from sklearn.linear_model import SGDClassifier

# نقوم بإنشاء 50 نقطة قابلة للفصل
X, Y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)

# تدريب النموذج
clf = SGDClassifier(loss="hinge", alpha=0.01, max_iter=200)

clf.fit(X, Y)

# رسم الخط، النقاط، وأقرب المتجهات إلى المستوى
xx = np.linspace(-1, 5, 10)
yy = np.linspace(-1, 5, 10)

X1, X2 = np.meshgrid(xx, yy)
Z = np.empty(X1.shape)
for (i, j), val in np.ndenumerate(X1):
    x1 = val
    x2 = X2[i, j]
    p = clf.decision_function([[x1, x2]])
    Z[i, j] = p[0]
levels = [-1.0, 0.0, 1.0]
linestyles = ["dashed", "solid", "dashed"]
colors = "k"
plt.contour(X1, X2, Z, levels, colors=colors, linestyles=linestyles)
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired, edgecolor="black", s=20)

plt.axis("tight")
plt.show()