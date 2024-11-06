"""
===================================
محول الأعمدة مع الأنواع المختلطة
===================================

.. currentmodule:: sklearn

يوضح هذا المثال كيفية تطبيق خطوط أنابيب مختلفة للمعالجة المسبقة واستخراج الميزات على مجموعات فرعية مختلفة من الميزات، باستخدام
:class:`~compose.ColumnTransformer`. هذا مفيد بشكل خاص في حالة مجموعات البيانات التي تحتوي على أنواع بيانات غير متجانسة، حيث قد نرغب في
قياس الميزات الرقمية وترميز الميزات الفئوية بنظام الترميز الثنائي.

في هذا المثال، يتم توحيد قياس البيانات الرقمية بعد إسناد القيم المتوسطة.
يتم ترميز البيانات الفئوية بنظام الترميز الثنائي عبر ``OneHotEncoder``، مما
يؤدي إلى إنشاء فئة جديدة للقيم المفقودة. نقوم أيضًا بتقليل الأبعاد
عن طريق تحديد الفئات باستخدام اختبار مربع كاي.

بالإضافة إلى ذلك، نعرض طريقتين مختلفتين لتوزيع الأعمدة على
أداة المعالجة المسبقة الخاصة: حسب أسماء الأعمدة وحسب أنواع بيانات الأعمدة.

أخيرًا، يتم دمج خط أنابيب المعالجة المسبقة في خط أنابيب التنبؤ الكامل
باستخدام :class:`~pipeline.Pipeline`، جنبًا إلى جنب مع نموذج تصنيف بسيط.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
import pandas as pd
from sklearn.compose import make_column_selector as selector
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

np.random.seed(0)

# %%
# تحميل البيانات من https://www.openml.org/d/40945
X, y = fetch_openml("titanic", version=1, as_frame=True, return_X_y=True)

# بدلاً من ذلك، يمكن الحصول على X و y مباشرةً من سمة الإطار:
# X = titanic.frame.drop('survived', axis=1)
# y = titanic.frame['survived']

# %%
# استخدام ``ColumnTransformer`` عن طريق تحديد العمود بالأسماء
#
# سندرب المصنف الخاص بنا بالمميزات التالية:
#
# المميزات الرقمية:
#
# * ``age``: عائم؛
# * ``fare``: عائم.
#
# المميزات الفئوية:
#
# * ``embarked``: فئات مشفرة كسلاسل ``{'C', 'S', 'Q'}``؛
# * ``sex``: فئات مشفرة كسلاسل ``{'female', 'male'}``؛
# * ``pclass``: أعداد صحيحة ترتيبية ``{1, 2, 3}``.
#
# نقوم بإنشاء خطوط أنابيب المعالجة المسبقة لكل من البيانات الرقمية والفئوية.
# لاحظ أنه يمكن معاملة ``pclass`` إما كميزة فئوية أو رقمية.

numeric_features = ["age", "fare"]
numeric_transformer = Pipeline(
    steps=[("imputer", SimpleImputer(strategy="median")),
           ("scaler", StandardScaler())]
)

categorical_features = ["embarked", "sex", "pclass"]
categorical_transformer = Pipeline(
    steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ("selector", SelectPercentile(chi2, percentile=50)),
    ]
)
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# %%
# إلحاق المصنف بخط أنابيب المعالجة المسبقة.
# الآن لدينا خط أنابيب تنبؤ كامل.
clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", LogisticRegression())]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))

# %%
# تمثيل HTML لـ ``Pipeline`` (عرض الرسم التخطيطي)
#
# عندما تتم طباعة ``Pipeline`` في دفتر ملاحظات jupyter، يتم عرض تمثيل HTML
# للمقدر:
clf

# %%
# استخدام ``ColumnTransformer`` عن طريق تحديد العمود حسب أنواع البيانات
#
# عند التعامل مع مجموعة بيانات منظفة، يمكن أن تكون المعالجة المسبقة تلقائية
# باستخدام أنواع بيانات العمود لتحديد ما إذا كان يجب معاملة العمود على أنه
# ميزة رقمية أو فئوية.
# :func:`sklearn.compose.make_column_selector` يعطي هذه الإمكانية.
# أولاً، لنحدد فقط مجموعة فرعية من الأعمدة لتبسيط
# مثالنا.

subset_feature = ["embarked", "sex", "pclass", "age", "fare"]
X_train, X_test = X_train[subset_feature], X_test[subset_feature]

# %%
# ثم، نتفحص المعلومات المتعلقة بكل نوع بيانات عمود.

X_train.info()

# %%
# يمكننا ملاحظة أن عمودي `embarked` و `sex` تم وضع علامة عليهما كأعمدة
# `category` عند تحميل البيانات باستخدام ``fetch_openml``. لذلك، يمكننا
# استخدام هذه المعلومات لتوزيع الأعمدة الفئوية على
# ``categorical_transformer`` والأعمدة المتبقية على
# ``numerical_transformer``.

# %%
# .. note:: عمليًا، سيتعين عليك التعامل بنفسك مع نوع بيانات العمود.
#    إذا كنت تريد اعتبار بعض الأعمدة على أنها `category`، فسيتعين عليك
#    تحويلها إلى أعمدة فئوية. إذا كنت تستخدم pandas، فيمكنك
#    الرجوع إلى وثائقهم المتعلقة بـ `البيانات الفئوية
#    <https://pandas.pydata.org/pandas-docs/stable/user_guide/categorical.html>`_.


preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, selector(dtype_exclude="category")),
        ("cat", categorical_transformer, selector(dtype_include="category")),
    ]
)
clf = Pipeline(
    steps=[("preprocessor", preprocessor),
           ("classifier", LogisticRegression())]
)


clf.fit(X_train, y_train)
print("model score: %.3f" % clf.score(X_test, y_test))
clf

# %%
# النتيجة الناتجة ليست تمامًا نفس النتيجة من خط الأنابيب السابق لأن المحدد القائم على dtype يعامل عمود ``pclass`` على أنه
# ميزة رقمية بدلاً من ميزة فئوية كما كان من قبل:

selector(dtype_exclude="category")(X_train)

# %%

selector(dtype_include="category")(X_train)


# %%
# استخدام خط أنابيب التنبؤ في بحث شبكي
#
# يمكن أيضًا إجراء البحث الشبكي على خطوات المعالجة المسبقة المختلفة
# المحددة في كائن ``ColumnTransformer``، جنبًا إلى جنب مع معلمات المصنف الفائقة
# كجزء من ``Pipeline``.
# سنبحث عن كل من استراتيجية الإسناد للمعالجة المسبقة الرقمية
# ومعلمة التنظيم للانحدار اللوجستي باستخدام
# :class:`~sklearn.model_selection.RandomizedSearchCV`. يختار بحث المعلمات الفائقة هذا عشوائيًا عددًا ثابتًا من إعدادات المعلمات
# التي تم تكوينها بواسطة `n_iter`. بدلاً من ذلك، يمكن للمرء استخدام
# :class:`~sklearn.model_selection.GridSearchCV` ولكن سيتم تقييم المنتج الديكارتي
# لمساحة المعلمات.


param_grid = {
    "preprocessor__num__imputer__strategy": ["mean", "median"],
    "preprocessor__cat__selector__percentile": [10, 30, 50, 70],
    "classifier__C": [0.1, 1.0, 10, 100],
}

search_cv = RandomizedSearchCV(clf, param_grid, n_iter=10, random_state=0)
search_cv

# %%
# يؤدي استدعاء 'fit' إلى تشغيل البحث المتقاطع للتحقق من صحة أفضل
# مزيج من المعلمات الفائقة:
#
search_cv.fit(X_train, y_train)

print("Best params:")
print(search_cv.best_params_)

# %%
# درجات التحقق المتقاطع الداخلي التي تم الحصول عليها بواسطة هذه المعلمات هي:
print(f"Internal CV score: {search_cv.best_score_:.3f}")

# %%
# يمكننا أيضًا فحص أفضل نتائج البحث الشبكي كإطار بيانات pandas:

cv_results = pd.DataFrame(search_cv.cv_results_)
cv_results = cv_results.sort_values("mean_test_score", ascending=False)
cv_results[
    [
        "mean_test_score",
        "std_test_score",
        "param_preprocessor__num__imputer__strategy",
        "param_preprocessor__cat__selector__percentile",
        "param_classifier__C",
    ]
].head(5)

# %%
# تم استخدام أفضل المعلمات الفائقة لإعادة ملاءمة نموذج نهائي على مجموعة التدريب الكاملة. يمكننا تقييم ذلك النموذج النهائي على بيانات الاختبار المحتجزة التي لم يتم استخدامها لضبط المعلمات الفائقة.
#
print(
    "accuracy of the best model from randomized search: "
    f"{search_cv.score(X_test, y_test):.3f}"
)
