"""
===========================
مطابقة متعامدة
===========================

استخدام مطابقة متعامدة لاستعادة إشارة متفرقة من قياس مشوش
مشفر بقاموس

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_sparse_coded_signal
from sklearn.linear_model import OrthogonalMatchingPursuit, OrthogonalMatchingPursuitCV

n_components, n_features = 512, 100
n_nonzero_coefs = 17

# توليد البيانات

# y = Xw
# |x|_0 = n_nonzero_coefs

y, X, w = make_sparse_coded_signal(
    n_samples=1,
    n_components=n_components,
    n_features=n_features,
    n_nonzero_coefs=n_nonzero_coefs,
    random_state=0,
)
X = X.T

(idx,) = w.nonzero()

# تشويه الإشارة النظيفة
y_noisy = y + 0.05 * np.random.randn(len(y))

# رسم الإشارة المتفرقة
plt.figure(figsize=(7, 7))
plt.subplot(4, 1, 1)
plt.xlim(0, 512)
plt.title("الإشارة المتفرقة")
plt.stem(idx, w[idx])

# رسم إعادة البناء الخالي من الضوضاء
omp = OrthogonalMatchingPursuit(n_nonzero_coefs=n_nonzero_coefs)
omp.fit(X, y)
coef = omp.coef_
(idx_r,) = coef.nonzero()
plt.subplot(4, 1, 2)
plt.xlim(0, 512)
plt.title("إعادة بناء الإشارة من القياسات الخالية من الضوضاء")
plt.stem(idx_r, coef[idx_r])

# رسم إعادة البناء المشوش
omp.fit(X, y_noisy)
coef = omp.coef_
(idx_r,) = coef.nonzero()
plt.subplot(4, 1, 3)
plt.xlim(0, 512)
plt.title("إعادة بناء الإشارة من القياسات المشوشة")
plt.stem(idx_r, coef[idx_r])

# رسم إعادة البناء المشوش مع عدد غير الصفري المحدد بواسطة CV
omp_cv = OrthogonalMatchingPursuitCV()
omp_cv.fit(X, y_noisy)
coef = omp_cv.coef_
(idx_r,) = coef.nonzero()
plt.subplot(4, 1, 4)
plt.xlim(0, 512)
plt.title("إعادة بناء الإشارة من القياسات المشوشة مع CV")
plt.stem(idx_r, coef[idx_r])

plt.subplots_adjust(0.06, 0.04, 0.94, 0.90, 0.20, 0.38)
plt.suptitle("استعادة الإشارة المتفرقة باستخدام مطابقة متعامدة", fontsize=16)
plt.show()