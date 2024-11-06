"""
===========================================
تقدير النموذج الخطي القوي باستخدام RANSAC
===========================================

في هذا المثال، نرى كيفية ملاءمة نموذج خطي بشكل قوي لبيانات معيبة باستخدام
خوارزمية :ref:`RANSAC <ransac_regression>`.

المُرَجِّع الخطي العادي حساس للقيم الشاذة، ويمكن للخط المناسب أن
يتحيز بسهولة بعيدًا عن العلاقة الأساسية الحقيقية للبيانات.

يقوم مُرَجِّع RANSAC تلقائيًا بتقسيم البيانات إلى قيم داخلية وخارجية،
ويتم تحديد الخط المناسب فقط من خلال القيم الداخلية المحددة.


"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn import datasets, linear_model

n_samples = 1000
n_outliers = 50


X, y, coef = datasets.make_regression(
    n_samples=n_samples,
    n_features=1,
    n_informative=1,
    noise=10,
    coef=True,
    random_state=0,
)

# إضافة بيانات شاذة
np.random.seed(0)
X[:n_outliers] = 3 + 0.5 * np.random.normal(size=(n_outliers, 1))
y[:n_outliers] = -3 + 10 * np.random.normal(size=n_outliers)

# ملاءمة الخط باستخدام جميع البيانات
lr = linear_model.LinearRegression()
lr.fit(X, y)

# ملاءمة النموذج الخطي بشكل قوي باستخدام خوارزمية RANSAC
ransac = linear_model.RANSACRegressor()
ransac.fit(X, y)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

# توقع بيانات النماذج المقدرة
line_X = np.arange(X.min(), X.max())[:, np.newaxis]
line_y = lr.predict(line_X)
line_y_ransac = ransac.predict(line_X)

# مقارنة المعاملات المقدرة
print("المعاملات المقدرة (الحقيقية، الانحدار الخطي، RANSAC):")
print(coef, lr.coef_, ransac.estimator_.coef_)

lw = 2
plt.scatter(
    X[inlier_mask], y[inlier_mask], color="yellowgreen", marker=".", label="Inliers"
)
plt.scatter(
    X[outlier_mask], y[outlier_mask], color="gold", marker=".", label="Outliers"
)
plt.plot(line_X, line_y, color="navy", linewidth=lw, label="Linear regressor")
plt.plot(
    line_X,
    line_y_ransac,
    color="cornflowerblue",
    linewidth=lw,
    label="RANSAC regressor",
)
plt.legend(loc="lower right")
plt.xlabel("Input")
plt.ylabel("Response")
plt.show()