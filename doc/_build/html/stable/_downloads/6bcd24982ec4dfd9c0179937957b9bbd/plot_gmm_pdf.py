"""
=============================
تقدير الكثافة لمزيج غاوسي
=============================

ارسم تقدير الكثافة لمزيج من غاوسيين. يتم توليد البيانات
من غاوسيين لهما مراكز ومصفوفات تباين مختلفة.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

from sklearn import mixture

n_samples = 300

# توليد عينة عشوائية، مكونان
np.random.seed(0)

# توليد بيانات كروية مركزها (20, 20)
shifted_gaussian = np.random.randn(n_samples, 2) + np.array([20, 20])

# توليد بيانات غاوسية ممدودة مركزها صفر
C = np.array([[0.0, -0.7], [3.5, 0.7]])
stretched_gaussian = np.dot(np.random.randn(n_samples, 2), C)

# دمج المجموعتين في مجموعة التدريب النهائية
X_train = np.vstack([shifted_gaussian, stretched_gaussian])

# ملاءمة نموذج مزيج غاوسي بمكونين
clf = mixture.GaussianMixture(n_components=2, covariance_type="full")
clf.fit(X_train)

# عرض الدرجات المتوقعة بواسطة النموذج كخريطة خطوط
x = np.linspace(-20.0, 30.0)
y = np.linspace(-20.0, 40.0)
X, Y = np.meshgrid(x, y)
XX = np.array([X.ravel(), Y.ravel()]).T
Z = -clf.score_samples(XX)
Z = Z.reshape(X.shape)

CS = plt.contour(
    X, Y, Z, norm=LogNorm(vmin=1.0, vmax=1000.0), levels=np.logspace(0, 3, 10)
)
CB = plt.colorbar(CS, shrink=0.8, extend="both")
plt.scatter(X_train[:, 0], X_train[:, 1], 0.8)

plt.title("Negative log-likelihood predicted by a GMM")
plt.axis("tight")
plt.show()