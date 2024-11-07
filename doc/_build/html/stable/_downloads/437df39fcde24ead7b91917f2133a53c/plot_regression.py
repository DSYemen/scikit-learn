"""
=======================
أقرب جيران الانحدار
=======================

توضيح حل مشكلة الانحدار
باستخدام أقرب جيران k- ونظام الاستيفاء للهدف باستخدام كل من الأوزان الثابتة والمركز الثقل.
"""
# %%
# توليد بيانات العينة
# --------------------
# هنا نقوم بتوليد بعض نقاط البيانات لاستخدامها في تدريب النموذج. كما نقوم بتوليد
# بيانات في النطاق الكامل لبيانات التدريب لتوضيح كيفية تفاعل النموذج
# في تلك المنطقة بالكامل.

import matplotlib.pyplot as plt
import numpy as np

from sklearn import neighbors

rng = np.random.RandomState(0)
X_train = np.sort(5 * rng.rand(40, 1), axis=0)
X_test = np.linspace(0, 5, 500)[:, np.newaxis]
y = np.sin(X_train).ravel()

# إضافة ضوضاء إلى الأهداف
y[::5] += 1 * (0.5 - np.random.rand(8))

# %%
# نموذج الانحدار المناسب
# --------------------
# هنا نقوم بتدريب نموذج وتوضيح كيفية تأثير الأوزان 'الموحدة' و 'المسافة'
# في التنبؤ بالقيم المتوقعة.

n_neighbors = 5

for i, weights in enumerate(["uniform", "distance"]):
    knn = neighbors.KNeighborsRegressor(n_neighbors, weights=weights)
    y_ = knn.fit(X_train, y).predict(X_test)

    plt.subplot(2, 1, i + 1)
    plt.scatter(X_train, y, color="darkorange", label="data")
    plt.plot(X_test, y_, color="navy", label="prediction")
    plt.axis("tight")
    plt.legend()
    plt.title("KNeighborsRegressor (k = %i, weights = '%s')" % (n_neighbors, weights))

plt.tight_layout()
plt.show()