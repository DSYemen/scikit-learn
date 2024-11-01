"""
=========================================================
ربط الأنابيب: ربط PCA والانحدار اللوجستي
=========================================================

يقوم PCA بتقليل الأبعاد بطريقة غير خاضعة للإشراف، بينما يقوم الانحدار اللوجستي بالتنبؤ.

نستخدم GridSearchCV لتعيين أبعاد PCA

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# تعريف خط أنابيب للبحث عن أفضل مزيج من اقتطاع PCA
# وانتظام المصنف.
pca = PCA()
# تعريف مقياس قياسي لتطبيع المدخلات
scaler = StandardScaler()

# تعيين التسامح إلى قيمة كبيرة لجعل المثال أسرع
logistic = LogisticRegression(max_iter=10000, tol=0.1)
pipe = Pipeline(steps=[("scaler", scaler),
                ("pca", pca), ("logistic", logistic)])

X_digits, y_digits = datasets.load_digits(return_X_y=True)
# يمكن تعيين معلمات خطوط الأنابيب باستخدام أسماء المعلمات المفصولة بـ '__':
param_grid = {
    "pca__n_components": [5, 15, 30, 45, 60],
    "logistic__C": np.logspace(-4, 4, 4),
}
search = GridSearchCV(pipe, param_grid, n_jobs=2)
search.fit(X_digits, y_digits)
print("Best parameter (CV score=%0.3f):" % search.best_score_)
print(search.best_params_)

# رسم طيف PCA
pca.fit(X_digits)

fig, (ax0, ax1) = plt.subplots(nrows=2, sharex=True, figsize=(6, 6))
ax0.plot(
    np.arange(1, pca.n_components_ + 1), pca.explained_variance_ratio_, "+", linewidth=2
)
ax0.set_ylabel("نسبة التباين الموضحة بواسطة PCA")

ax0.axvline(
    search.best_estimator_.named_steps["pca"].n_components,
    linestyle=":",
    label="n_components المختارة",
)
ax0.legend(prop=dict(size=12))

# لكل عدد من المكونات، ابحث عن أفضل نتائج المصنف
components_col = "param_pca__n_components"
is_max_test_score = pl.col("mean_test_score") == pl.col(
    "mean_test_score").max()
best_clfs = (
    pl.LazyFrame(search.cv_results_)
    .filter(is_max_test_score.over(components_col))
    .unique(components_col)
    .sort(components_col)
    .collect()
)
ax1.errorbar(
    best_clfs[components_col],
    best_clfs["mean_test_score"],
    yerr=best_clfs["std_test_score"],
)
ax1.set_ylabel("دقة التصنيف (val)")
ax1.set_xlabel("n_components")

plt.xlim(-1, 70)

plt.tight_layout()
plt.show()
