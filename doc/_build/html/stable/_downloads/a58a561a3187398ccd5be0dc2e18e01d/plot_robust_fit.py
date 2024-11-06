"""
================
تقدير خطي قوي
================

هنا يتم ملاءمة دالة الجيب مع متعددة حدود من الدرجة 3، للقيم
قريبة من الصفر.

يتم عرض الملاءمة القوية في مواقف مختلفة:

- لا توجد أخطاء في القياس، فقط أخطاء في النمذجة (ملاءمة دالة الجيب مع
  متعددة حدود)

- أخطاء في قياس X

- أخطاء في قياس y

يتم استخدام الانحراف المتوسط المطلق للبيانات الجديدة غير الفاسدة للحكم
على جودة التوقع.

ما يمكننا رؤيته هو:

- RANSAC جيد للبيانات الشاذة القوية في اتجاه y

- TheilSen جيد للبيانات الشاذة الصغيرة، في الاتجاهين X و y، ولكن لديه
  نقطة كسر فوقها يؤدي إلى أداء أسوأ من OLS.

- قد لا يتم مقارنة درجات HuberRegressor مباشرة مع كل من TheilSen
  و RANSAC لأنه لا يحاول تصفية البيانات الشاذة تمامًا
  ولكن تقليل تأثيرها.
"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.linear_model import (
    HuberRegressor,
    LinearRegression,
    RANSACRegressor,
    TheilSenRegressor,
)
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

np.random.seed(42)

X = np.random.normal(size=400)
y = np.sin(X)
# تأكد من أن X ثنائي الأبعاد
X = X[:, np.newaxis]

X_test = np.random.normal(size=200)
y_test = np.sin(X_test)
X_test = X_test[:, np.newaxis]

y_errors = y.copy()
y_errors[::3] = 3

X_errors = X.copy()
X_errors[::3] = 3

y_errors_large = y.copy()
y_errors_large[::3] = 10

X_errors_large = X.copy()
X_errors_large[::3] = 10

estimators = [
    ("OLS", LinearRegression()),
    ("Theil-Sen", TheilSenRegressor(random_state=42)),
    ("RANSAC", RANSACRegressor(random_state=42)),
    ("HuberRegressor", HuberRegressor()),
]
colors = {
    "OLS": "turquoise",
    "Theil-Sen": "gold",
    "RANSAC": "lightgreen",
    "HuberRegressor": "black",
}
linestyle = {"OLS": "-", "Theil-Sen": "-.", "RANSAC": "--", "HuberRegressor": "--"}
lw = 3

x_plot = np.linspace(X.min(), X.max())
for title, this_X, this_y in [
    ("Modeling Errors Only", X, y),
    ("Corrupt X, Small Deviants", X_errors, y),
    ("Corrupt y, Small Deviants", X, y_errors),
    ("Corrupt X, Large Deviants", X_errors_large, y),
    ("Corrupt y, Large Deviants", X, y_errors_large),
]:
    plt.figure(figsize=(5, 4))
    plt.plot(this_X[:, 0], this_y, "b+")

    for name, estimator in estimators:
        model = make_pipeline(PolynomialFeatures(3), estimator)
        model.fit(this_X, this_y)
        mse = mean_squared_error(model.predict(X_test), y_test)
        y_plot = model.predict(x_plot[:, np.newaxis])
        plt.plot(
            x_plot,
            y_plot,
            color=colors[name],
            linestyle=linestyle[name],
            linewidth=lw,
            label="%s: error = %.3f" % (name, mse),
        )

    legend_title = "Error of Mean\nAbsolute Deviation\nto Non-corrupt Data"
    legend = plt.legend(
        loc="upper right", frameon=False, title=legend_title, prop=dict(size="x-small")
    )
    plt.xlim(-4, 10.2)
    plt.ylim(-2, 10.2)
    plt.title(title)
plt.show()