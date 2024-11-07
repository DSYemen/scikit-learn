"""
==============================================================
الكشف عن القيم الشاذة باستخدام عامل الانحراف المحلي (LOF)
==============================================================

خوارزمية عامل الانحراف المحلي (LOF) هي طريقة غير خاضعة للإشراف للكشف عن الانحرافات
والتي تحسب انحراف الكثافة المحلية لنقطة بيانات معينة فيما يتعلق بجيرانها.
تعتبر هذه الخوارزمية العينات التي لها كثافة أقل بكثير من جيرانها كقيم شاذة.
يوضح هذا المثال كيفية استخدام LOF للكشف عن القيم الشاذة، وهو الاستخدام الافتراضي
لهذا المقدر في مكتبة ساي كيت ليرن (scikit-learn). تجدر الإشارة إلى أنه عند استخدام
LOF للكشف عن القيم الشاذة، لا تتوفر له طرق 'predict' و 'decision_function' و
'score_samples'. راجع :ref:`دليل المستخدم <outlier_detection>` للحصول على التفاصيل حول
الفرق بين الكشف عن القيم الشاذة والكشف عن البيانات الجديدة وكيفية استخدام LOF للكشف
عن البيانات الجديدة.

يتم ضبط عدد الجيران المعتبرين (البارامتر 'n_neighbors') عادةً 1) أكبر من الحد الأدنى
لعدد العينات التي يجب أن يحتويها التجمع، بحيث يمكن أن تكون العينات الأخرى قيمًا
شاذة محلية بالنسبة لهذا التجمع، و2) أصغر من الحد الأقصى لعدد العينات القريبة التي
يمكن أن تكون قيمًا شاذة محلية. في الممارسة العملية، هذه المعلومات غير متوفرة
عادةً، ويبدو أن أخذ 'n_neighbors=20' يعمل بشكل جيد بشكل عام.

"""

# المؤلفون: مطوري ساي كيت ليرن (scikit-learn)
# معرف الترخيص: BSD-3-Clause

# %%
# توليد البيانات مع القيم الشاذة
# ---------------------------

# %%
import numpy as np

np.random.seed(42)

X_inliers = 0.3 * np.random.randn(100, 2)
X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.r_[X_inliers, X_outliers]

n_outliers = len(X_outliers)
ground_truth = np.ones(len(X), dtype=int)
ground_truth[-n_outliers:] = -1

# %%
# ملاءمة النموذج للكشف عن القيم الشاذة (الافتراضي)
# ---------------------------------------------
#
# استخدم 'fit_predict' لحساب العلامات المتوقعة لعينات التدريب
# (عندما يتم استخدام LOF للكشف عن القيم الشاذة، لا يمتلك المقدر طرق 'predict'
# و 'decision_function' و 'score_samples').

from sklearn.neighbors import LocalOutlierFactor

clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred = clf.fit_predict(X)
n_errors = (y_pred != ground_truth).sum()
X_scores = clf.negative_outlier_factor_

# %%
# رسم النتائج
# ------------

# %%

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection


def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])


plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
# plot circles with radius proportional to the outlier scores
radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
scatter = plt.scatter(
    X[:, 0],
    X[:, 1],
    s=1000 * radius,
    edgecolors="r",
    facecolors="none",
    label="Outlier scores",
)
plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.xlabel("prediction errors: %d" % (n_errors))
plt.legend(
    handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
)
plt.title("Local Outlier Factor (LOF)")
plt.show()
