"""
======================================
انحدار شجرة القرار مع AdaBoost
======================================

شجرة القرار معززة باستخدام خوارزمية AdaBoost.R2 [1]_ على مجموعة بيانات جيبية أحادية البعد مع كمية صغيرة من الضوضاء الغاوسية.
يتم مقارنة 299 دفعة (300 شجرة قرار) مع منظم شجرة قرار واحد. مع زيادة عدد الدفعات، يمكن لمنظم الانحدار أن يلائم المزيد من التفاصيل.

راجع :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py` لمثال يبرز فوائد استخدام نماذج الانحدار الأكثر كفاءة مثل :class:`~ensemble.HistGradientBoostingRegressor`.

.. [1] `H. Drucker, "Improving Regressors using Boosting Techniques", 1997.
        <https://citeseerx.ist.psu.edu/doc_view/pid/8d49e2dedb817f2c3330e74b63c5fc86d2399ce3>`_

# %%
# إعداد البيانات
# ------------------
# أولاً، نقوم بإعداد بيانات وهمية بعلاقة جيبية وبعض الضوضاء الغاوسية.
"""
# المؤلفون: مطوري سكايت-ليرن
# معرف الترخيص: BSD-3-Clause

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
import numpy as np

rng = np.random.RandomState(1)
X = np.linspace(0, 6, 100)[:, np.newaxis]
y = np.sin(X).ravel() + np.sin(6 * X).ravel() + rng.normal(0, 0.1, X.shape[0])

# %%
# التدريب والتنبؤ باستخدام منظمي شجرة القرار و AdaBoost
# -----------------------------------------------------------------
# الآن، نقوم بتعريف المنظمين وتناسبهم مع البيانات.
# ثم نتوقع على تلك البيانات نفسها لنرى مدى ملاءمتها.
# المنظم الأول هو `DecisionTreeRegressor` مع `max_depth=4`.
# المنظم الثاني هو `AdaBoostRegressor` مع `DecisionTreeRegressor`
# ب `max_depth=4` كمتعلم أساسي وسيتم بناؤه مع `n_estimators=300`
# من تلك المتعلمات الأساسية.


regr_1 = DecisionTreeRegressor(max_depth=4)

regr_2 = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=4), n_estimators=300, random_state=rng
)

regr_1.fit(X, y)
regr_2.fit(X, y)

y_1 = regr_1.predict(X)
y_2 = regr_2.predict(X)

# %%
# رسم النتائج
# --------------------
# أخيرًا، نرسم مدى ملاءمة منظمينا،
# منظم شجرة القرار المفرد ومنظم AdaBoost، للبيانات.


colors = sns.color_palette("colorblind")

plt.figure()
plt.scatter(X, y, color=colors[0], label="training samples")
plt.plot(X, y_1, color=colors[1], label="n_estimators=1", linewidth=2)
plt.plot(X, y_2, color=colors[2], label="n_estimators=300", linewidth=2)
plt.xlabel("data")
plt.ylabel("target")
plt.title("Boosted Decision Tree Regression")
plt.legend()
plt.show()
