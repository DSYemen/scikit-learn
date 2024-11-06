"""
============================================
Model-based and sequential feature selection
============================================

يوضح هذا المثال ويقارن نهجين لاختيار الميزات:
:class:`~sklearn.feature_selection.SelectFromModel` الذي يعتمد على أهمية
الميزات، و :class:`~sklearn.feature_selection.SequentialFeatureSelector` الذي
يعتمد على نهج جشع.

نستخدم مجموعة بيانات السكري، والتي تتكون من 10 ميزات تم جمعها من 442
مريضًا بالسكري.

Authors: `Manoj Kumar <mks542@nyu.edu>`_,
`Maria Telenczuk <https://github.com/maikia>`_, Nicolas Hug.

License: BSD 3 clause

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# تحميل البيانات
# ----------------
#
# نقوم أولاً بتحميل مجموعة بيانات السكري المتاحة من داخل
# scikit-learn، ونطبع وصفها:
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
print(diabetes.DESCR)

# %%
# أهمية الميزات من المعاملات
# ------------------------------------
#
# للحصول على فكرة عن أهمية الميزات، سنستخدم
# مقدّر :class:`~sklearn.linear_model.RidgeCV`. تعتبر الميزات ذات
# أعلى قيمة مطلقة لـ `coef_` هي الأكثر أهمية.
# يمكننا ملاحظة المعاملات مباشرة دون الحاجة إلى تغيير مقياسها (أو
# تغيير مقياس البيانات) لأنه من الوصف أعلاه، نعلم أن الميزات
# قد تم توحيدها بالفعل.
# للحصول على مثال أكثر اكتمالاً عن تفسيرات معاملات النماذج
# الخطية، يمكنك الرجوع إلى
# :ref:`sphx_glr_auto_examples_inspection_plot_linear_model_coefficient_interpretation.py`.  # noqa: E501
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import RidgeCV

ridge = RidgeCV(alphas=np.logspace(-6, 6, num=5)).fit(X, y)
importance = np.abs(ridge.coef_)
feature_names = np.array(diabetes.feature_names)
plt.bar(height=importance, x=feature_names)
plt.title("أهمية الميزات عبر المعاملات")
plt.show()

# %%
# تحديد الميزات بناءً على الأهمية
# --------------------------------------
#
# الآن نريد تحديد الميزتين الأكثر أهمية وفقًا
# للمعاملات. :class:`~sklearn.feature_selection.SelectFromModel`
# مخصصة لذلك فقط. :class:`~sklearn.feature_selection.SelectFromModel`
# تقبل معلمة `threshold` وستحدد الميزات التي تكون أهميتها
# (المحددة بواسطة المعاملات) أعلى من هذه العتبة.
#
# نظرًا لأننا نريد تحديد ميزتين فقط، فسنقوم بتعيين هذه العتبة أعلى
# بقليل من معامل ثالث أهم ميزة.
from time import time

from sklearn.feature_selection import SelectFromModel

threshold = np.sort(importance)[-3] + 0.01

tic = time()
sfm = SelectFromModel(ridge, threshold=threshold).fit(X, y)
toc = time()
print(f"الميزات المحددة بواسطة SelectFromModel: {feature_names[sfm.get_support()]}")
print(f"تم في {toc - tic:.3f}s")

# %%
# تحديد الميزات مع اختيار الميزات المتسلسل
# ----------------------------------------------------
#
# هناك طريقة أخرى لتحديد الميزات وهي استخدام
# :class:`~sklearn.feature_selection.SequentialFeatureSelector`
# (SFS). SFS هو إجراء جشع حيث، في كل تكرار، نختار أفضل
# ميزة جديدة لإضافتها إلى ميزاتنا المحددة بناءً على درجة التحقق المتبادل.
# أي أننا نبدأ بصفر ميزات ونختار أفضل ميزة واحدة بأعلى درجة.
# يتم تكرار الإجراء حتى نصل إلى العدد المطلوب من الميزات المحددة.
#
# يمكننا أيضًا الانتقال في الاتجاه المعاكس (SFS للخلف)، *أي* البدء بجميع
# الميزات واختيار الميزات بشكل جشع لإزالتها واحدة تلو الأخرى. نوضح
# كلا النهجين هنا.


from sklearn.feature_selection import SequentialFeatureSelector

tic_fwd = time()
sfs_forward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="forward"
).fit(X, y)
toc_fwd = time()

tic_bwd = time()
sfs_backward = SequentialFeatureSelector(
    ridge, n_features_to_select=2, direction="backward"
).fit(X, y)
toc_bwd = time()

print(
    "الميزات المحددة بواسطة الاختيار المتسلسل للأمام: "
    f"{feature_names[sfs_forward.get_support()]}"
)
print(f"تم في {toc_fwd - tic_fwd:.3f}s")
print(
    "الميزات المحددة بواسطة الاختيار المتسلسل للخلف: "
    f"{feature_names[sfs_backward.get_support()]}"
)
print(f"تم في {toc_bwd - tic_bwd:.3f}s")

# %%
# من المثير للاهتمام أن الاختيار الأمامي والخلفي قد حددا نفس مجموعة
# الميزات. بشكل عام، ليس هذا هو الحال وستؤدي الطريقتان إلى
# نتائج مختلفة.
#
# نلاحظ أيضًا أن الميزات التي حددها SFS تختلف عن تلك التي حددتها
# أهمية الميزات: يحدد SFS `bmi` بدلاً من `s1`. يبدو هذا منطقيًا،
# نظرًا لأن `bmi` تتوافق مع ثالث أهم ميزة وفقًا للمعاملات. إنه أمر
# رائع للغاية بالنظر إلى أن SFS لا يستخدم المعاملات على الإطلاق.
#
# في الختام، تجدر الإشارة إلى أن
# :class:`~sklearn.feature_selection.SelectFromModel` أسرع بكثير
# من SFS. في الواقع، :class:`~sklearn.feature_selection.SelectFromModel`
# تحتاج فقط إلى ملاءمة نموذج مرة واحدة، بينما يحتاج SFS إلى التحقق المتبادل
# للعديد من النماذج المختلفة لكل تكرار. ومع ذلك، يعمل SFS مع أي نموذج،
# بينما تتطلب :class:`~sklearn.feature_selection.SelectFromModel` أن يعرض
# المقدّر الأساسي سمة `coef_` أو سمة `feature_importances_`.
# يكون SFS الأمامي أسرع من SFS الخلفي لأنه يحتاج فقط إلى إجراء
# `n_features_to_select = 2` تكرار، بينما يحتاج SFS الخلفي إلى إجراء
# `n_features - n_features_to_select = 8` تكرار.
#
# استخدام قيم التسامح السلبية
# -------------------------------
#
# :class:`~sklearn.feature_selection.SequentialFeatureSelector` يمكن استخدامها
# لإزالة الميزات الموجودة في مجموعة البيانات وإرجاع مجموعة فرعية
# أصغر من الميزات الأصلية مع `direction="backward"` وقيمة سالبة لـ `tol`.
#
# نبدأ بتحميل مجموعة بيانات سرطان الثدي، والتي تتكون من 30 ميزة
# مختلفة و 569 عينة.
import numpy as np

from sklearn.datasets import load_breast_cancer

breast_cancer_data = load_breast_cancer()
X, y = breast_cancer_data.data, breast_cancer_data.target
feature_names = np.array(breast_cancer_data.feature_names)
print(breast_cancer_data.DESCR)


# %%
# سنستخدم مقدّر :class:`~sklearn.linear_model.LogisticRegression`
# مع :class:`~sklearn.feature_selection.SequentialFeatureSelector`
# لإجراء اختيار الميزات.
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

for tol in [-1e-2, -1e-3, -1e-4]:
    start = time()
    feature_selector = SequentialFeatureSelector(
        LogisticRegression(),
        n_features_to_select="auto",
        direction="backward",
        scoring="roc_auc",
        tol=tol,
        n_jobs=2,
    )
    model = make_pipeline(StandardScaler(), feature_selector, LogisticRegression())
    model.fit(X, y)
    end = time()
    print(f"\ntol: {tol}")
    print(f"الميزات المحددة: {feature_names[model[1].get_support()]}")
    print(f"درجة ROC AUC: {roc_auc_score(y, model.predict_proba(X)[:, 1]):.3f}")
    print(f"تم في {end - start:.3f}s")

# %%
# يمكننا أن نرى أن عدد الميزات المحددة يميل إلى الزيادة مع اقتراب القيم
# السالبة لـ `tol` من الصفر. يقل الوقت المستغرق لاختيار الميزات أيضًا
# مع اقتراب قيم `tol` من الصفر.
