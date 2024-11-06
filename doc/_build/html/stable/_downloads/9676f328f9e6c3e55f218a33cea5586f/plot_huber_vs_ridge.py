"""
=======================================================
مقارنة بين HuberRegressor و Ridge على مجموعة بيانات تحتوي على قيم شاذة قوية
=======================================================

قم بضبط نموذج Ridge و HuberRegressor على مجموعة بيانات تحتوي على قيم شاذة.

يوضح المثال أن التنبؤات في نموذج Ridge تتأثر بشدة
بالقيم الشاذة الموجودة في مجموعة البيانات. أما نموذج HuberRegressor فهو أقل
تأثراً بالقيم الشاذة حيث يستخدم دالة خسارة خطية لهذه القيم.
مع زيادة معامل إبسيلون في نموذج HuberRegressor، تقترب دالة القرار
من تلك الخاصة بنموذج Ridge.

"""

# المؤلفون: مطوري مكتبة ساي كيت ليرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, Ridge

# إنشاء بيانات تجريبية.
rng = np.random.RandomState(0)
X, y = make_regression(
    n_samples=20, n_features=1, random_state=0, noise=4.0, bias=100.0
)

# إضافة أربع قيم شاذة قوية إلى مجموعة البيانات.
X_outliers = rng.normal(0, 0.5, size=(4, 1))
y_outliers = rng.normal(0, 2.0, size=4)
X_outliers[:2, :] += X.max() + X.mean() / 4.0
X_outliers[2:, :] += X.min() - X.mean() / 4.0
y_outliers[:2] += y.min() - y.mean() / 4.0
y_outliers[2:] += y.max() + y.mean() / 4.0
X = np.vstack((X, X_outliers))
y = np.concatenate((y, y_outliers))
plt.plot(X, y, "b.")

# ضبط نموذج HuberRegressor على سلسلة من قيم إبسيلون.
colors = ["r-", "b-", "y-", "m-"]

x = np.linspace(X.min(), X.max(), 7)
epsilon_values = [1, 1.5, 1.75, 1.9]
for k, epsilon in enumerate(epsilon_values):
    huber = HuberRegressor(alpha=0.0, epsilon=epsilon)
    huber.fit(X, y)
    coef_ = huber.coef_ * x + huber.intercept_
    plt.plot(x, coef_, colors[k], label="huber loss, %s" % epsilon)

# ضبط نموذج Ridge للمقارنة مع نموذج HuberRegressor.
ridge = Ridge(alpha=0.0, random_state=0)
ridge.fit(X, y)
coef_ridge = ridge.coef_
coef_ = ridge.coef_ * x + ridge.intercept_
plt.plot(x, coef_, "g-", label="ridge regression")

plt.title("مقارنة بين HuberRegressor و Ridge")
plt.xlabel("X")
plt.ylabel("y")
plt.legend(loc=0)
plt.show()