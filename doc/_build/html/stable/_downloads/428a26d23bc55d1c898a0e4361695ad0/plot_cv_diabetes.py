"""
===============================================
التحقق المتقاطع على تمرين مجموعة بيانات مرض السكري
===============================================

تمرين تعليمي يستخدم التحقق المتقاطع مع النماذج الخطية.

يتم استخدام هذا التمرين في جزء :ref:`cv_estimators_tut` من
قسم :ref:`model_selection_tut` من :ref:`stat_learn_tut_index`.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# تحميل مجموعة البيانات وتطبيق GridSearchCV
# -----------------------------------
import matplotlib.pyplot as plt
import numpy as np

from sklearn import datasets
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV

X, y = datasets.load_diabetes(return_X_y=True)
X = X[:150]
y = y[:150]

lasso = Lasso(random_state=0, max_iter=10000)
alphas = np.logspace(-4, -0.5, 30)

tuned_parameters = [{"alpha": alphas}]
n_folds = 5

clf = GridSearchCV(lasso, tuned_parameters, cv=n_folds, refit=False)
clf.fit(X, y)
scores = clf.cv_results_["mean_test_score"]
scores_std = clf.cv_results_["std_test_score"]

# %%
# رسم خطوط الخطأ التي توضح +/- أخطاء قياسية للنتائج
# ------------------------------------------------------

plt.figure().set_size_inches(8, 6)
plt.semilogx(alphas, scores)

std_error = scores_std / np.sqrt(n_folds)

plt.semilogx(alphas, scores + std_error, "b--")
plt.semilogx(alphas, scores - std_error, "b--")

# alpha=0.2 يتحكم في شفافية لون التعبئة
plt.fill_between(alphas, scores + std_error, scores - std_error, alpha=0.2)

plt.ylabel("نتيجة CV +/- الخطأ القياسي")
plt.xlabel("alpha")
plt.axhline(np.max(scores), linestyle="--", color=".5")
plt.xlim([alphas[0], alphas[-1]])

# %%
# مكافأة: ما مدى ثقتك في اختيار alpha؟
# -----------------------------------------------------

# للإجابة على هذا السؤال، نستخدم كائن LassoCV الذي يضبط معلمة alpha
# تلقائيًا من البيانات عن طريق التحقق المتقاطع الداخلي (أي أنه
# ينفذ التحقق المتقاطع على بيانات التدريب التي يتلقاها).
# نستخدم التحقق المتقاطع الخارجي لمعرفة مدى اختلاف قيم alpha التي تم
# الحصول عليها تلقائيًا عبر طيات التحقق المتقاطع المختلفة.

from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

lasso_cv = LassoCV(alphas=alphas, random_state=0, max_iter=10000)
k_fold = KFold(3)

print("إجابة السؤال الإضافي:", "ما مدى ثقتك في اختيار alpha؟")
print()
print("معلمات Alpha التي تزيد من درجة التعميم على مجموعات فرعية مختلفة")
print("من البيانات:")
for k, (train, test) in enumerate(k_fold.split(X, y)):
    lasso_cv.fit(X[train], y[train])
    print(
        "[طي {0}] alpha: {1:.5f}, النتيجة: {2:.5f}".format(
            k, lasso_cv.alpha_, lasso_cv.score(X[test], y[test])
        )
    )
print()
print("الإجابة: ليست كبيرة جدًا نظرًا لأننا حصلنا على قيم alpha مختلفة لمجموعات فرعية مختلفة")
print("من البيانات، علاوة على ذلك، تختلف الدرجات لهذه القيم alpha")
print("بشكل كبير.")

plt.show()


