"""
==========================================================
مثال على خط أنابيب لاستخراج ميزات النص وتقييمها
==========================================================

مجموعة البيانات المستخدمة في هذا المثال هي :ref:`20newsgroups_dataset` والتي سيتم
تنزيلها تلقائيًا وتخزينها مؤقتًا وإعادة استخدامها لمثال تصنيف المستند.

في هذا المثال، نقوم بضبط معلمات نموذج معين باستخدام
:class:`~sklearn.model_selection.RandomizedSearchCV`. لمشاهدة أداء بعض المصنفات الأخرى، راجع
:ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`
دفتر الملاحظات.
"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

# %%
# تحميل البيانات
# ------------
# نقوم بتحميل فئتين من مجموعة التدريب. يمكنك ضبط عدد الفئات عن طريق إضافة أسمائها إلى القائمة أو تعيين `categories=None` عند
# استدعاء محمل مجموعة البيانات :func:`~sklearn.datasets.fetch_20newsgroups` للحصول
# على 20 منها.

from sklearn.datasets import fetch_20newsgroups

categories = [
    "alt.atheism",
    "talk.religion.misc",
]

data_train = fetch_20newsgroups(
    subset="train",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

data_test = fetch_20newsgroups(
    subset="test",
    categories=categories,
    shuffle=True,
    random_state=42,
    remove=("headers", "footers", "quotes"),
)

print(f"تحميل مجموعة بيانات 20 newsgroups لـ {len(data_train.target_names)} فئات:")
print(data_train.target_names)
print(f"{len(data_train.data)} وثائق")

# %%
# خط أنابيب مع ضبط المعلمات
# -----------------------------------
#
# نحن نحدد خط أنابيب يجمع بين مستخرج ميزات النص مع مصنف بسيط
# ولكن فعال لتصنيف النص.

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

pipeline = Pipeline(
    [
        ("vect", TfidfVectorizer()),
        ("clf", ComplementNB()),
    ]
)
pipeline

# %%
# نحن نحدد شبكة من المعلمات ليتم استكشافها بواسطة
# :class:`~sklearn.model_selection.RandomizedSearchCV`. استخدام
# :class:`~sklearn.model_selection.GridSearchCV` بدلاً من ذلك سيستكشف جميع
# المجموعات الممكنة على الشبكة، والتي يمكن أن تكون مكلفة في الحساب، في حين أن
# المعلمة `n_iter` من :class:`~sklearn.model_selection.RandomizedSearchCV`
# تتحكم في عدد المجموعات العشوائية المختلفة التي يتم تقييمها. لاحظ
# أن تعيين `n_iter` أكبر من عدد المجموعات الممكنة في
# الشبكة سيؤدي إلى تكرار المجموعات التي تم استكشافها بالفعل. نبحث عن
# أفضل مجموعة من المعلمات لكل من استخراج الميزات (`vect__`) والمصنف (`clf__`).

import numpy as np

parameter_grid = {
    "vect__max_df": (0.2, 0.4, 0.6, 0.8, 1.0),
    "vect__min_df": (1, 3, 5, 10),
    "vect__ngram_range": ((1, 1), (1, 2)),  # كلمات مفردة أو ثنائية
    "vect__norm": ("l1", "l2"),
    "clf__alpha": np.logspace(-6, 6, 13),
}

# %%
# في هذه الحالة، `n_iter=40` ليس بحثًا شاملًا لشبكة المعلمات. في الواقع، سيكون من المثير للاهتمام زيادة المعلمة `n_iter`
# للحصول على تحليل أكثر إفادة. ونتيجة لذلك، يزيد وقت الحساب. يمكننا تقليله عن طريق الاستفادة من التوازي على
# تقييم مجموعات المعلمات عن طريق زيادة عدد وحدات المعالجة المركزية المستخدمة
# عبر المعلمة `n_jobs`.

from pprint import pprint

from sklearn.model_selection import RandomizedSearchCV

random_search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=parameter_grid,
    n_iter=40,
    random_state=0,
    n_jobs=2,
    verbose=1,
)

print("أداء البحث الشبكي...")
print("معلمات ليتم تقييمها:")
pprint(parameter_grid)

# %%
from time import time

t0 = time()
random_search.fit(data_train.data, data_train.target)
print(f"تم الانتهاء في {time() - t0:.3f}s")

# %%
print("أفضل مجموعة من المعلمات التي تم العثور عليها:")
best_parameters = random_search.best_estimator_.get_params()
for param_name in sorted(parameter_grid.keys()):
    print(f"{param_name}: {best_parameters[param_name]}")

# %%
test_accuracy = random_search.score(data_test.data, data_test.target)
print(
    "دقة أفضل المعلمات باستخدام CV الداخلي لـ "
    f"البحث العشوائي: {random_search.best_score_:.3f}"
)
print(f"الدقة على مجموعة الاختبار: {test_accuracy:.3f}")

# %%
# البادئات `vect` و `clf` مطلوبة لتجنب الغموض المحتمل في
# خط الأنابيب، ولكنها غير ضرورية لعرض النتائج. بسبب
# هذا، نحن نحدد دالة ستعيد تسمية المعلمات التي تم ضبطها وتحسين قابلية القراءة.

import pandas as pd


def shorten_param(param_name):
    """إزالة بادئات المكونات في param_name."""
    if "__" in param_name:
        return param_name.rsplit("__", 1)[1]
    return param_name


cv_results = pd.DataFrame(random_search.cv_results_)
cv_results = cv_results.rename(shorten_param, axis=1)

# %%
# يمكننا استخدام `plotly.express.scatter
# <https://plotly.com/python-api-reference/generated/plotly.express.scatter.html>`_
# لعرض المقايضة بين وقت التسجيل ومتوسط درجة الاختبار (أي "درجة CV"). تمرير المؤشر فوق نقطة معينة يعرض المعلمات
# المقابلة. أشرطة الخطأ تقابل انحرافًا معياريًا واحدًا كما تم حسابه في
# الطيات المختلفة للتحقق المتقاطع.

import plotly.express as px

param_names = [shorten_param(name) for name in parameter_grid.keys()]
labels = {
    "mean_score_time": "وقت درجة CV (ثانية)",
    "mean_test_score": "درجة CV (الدقة)",
}
fig = px.scatter(
    cv_results,
    x="mean_score_time",
    y="mean_test_score",
    error_x="std_score_time",
    error_y="std_test_score",
    hover_data=param_names,
    labels=labels,
)
fig.update_layout(
    title={
        "text": "المقايضة بين وقت التسجيل ومتوسط درجة الاختبار",
        "y": 0.95,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)
fig

# %%
# لاحظ أن مجموعة النماذج في الركن العلوي الأيسر من الرسم البياني لها
# أفضل مقايضة بين الدقة ووقت التسجيل. في هذه الحالة، يؤدي استخدام
# الكلمات الثنائية إلى زيادة وقت التسجيل المطلوب دون تحسين دقة خط الأنابيب بشكل كبير.
#
# .. note:: للحصول على مزيد من المعلومات حول كيفية تخصيص ضبط تلقائي لتحقيق أقصى قدر من الدقة وتقليل وقت التسجيل، راجع دفتر الملاحظات
#    المثال: :ref:`sphx_glr_auto_examples_model_selection_plot_grid_search_digits.py`.
#
# يمكننا أيضًا استخدام `plotly.express.parallel_coordinates
# <https://plotly.com/python-api-reference/generated/plotly.express.parallel_coordinates.html>`_
# لعرض متوسط درجة الاختبار كدالة للمعلمات التي تم ضبطها. يساعد هذا في العثور على التفاعلات بين أكثر من
# معلمتين وتوفير الحدس حول أهميتها لتحسين أداء خط الأنابيب.
#
# نطبق تحويل `math.log10` على محور "alpha" لنشر النطاق النشط وتحسين
# قابلية قراءة الرسم البياني. يتم فهم القيمة :math:`x` على
# المحور المذكور على أنها :math:`10^x`.

import math

column_results = param_names + ["mean_test_score", "mean_score_time"]

transform_funcs = dict.fromkeys(column_results, lambda x: x)
# استخدام مقياس لوغاريتمي لـ alpha
transform_funcs["alpha"] = math.log10
# يتم تعيين المعايير L1 إلى الفهرس 1، والمعايير L2 إلى الفهرس 2
transform_funcs["norm"] = lambda x: 2 if x == "l2" else 1
# يتم تعيين الكلمات المفردة إلى الفهرس 1 والكلمات الثنائية إلى الفهرس 2
transform_funcs["ngram_range"] = lambda x: x[1]

fig = px.parallel_coordinates(
    cv_results[column_results].apply(transform_funcs),
    color="mean_test_score",
    color_continuous_scale=px.colors.sequential.Viridis_r,
    labels=labels,
)
fig.update_layout(
    title={
        "text": "رسم تنسيق متوازي لخط أنابيب مصنف النص",
        "y": 0.99,
        "x": 0.5,
        "xanchor": "center",
        "yanchor": "top",
    }
)
fig

# %%
# يعرض رسم التنسيق المتوازي قيم المعلمات على
# أعمدة مختلفة في حين يتم ترميز مقياس الأداء بالألوان. من الممكن
# تحديد نطاق من النتائج عن طريق النقر والضغط على أي محور من
# رسم التنسيق المتوازي. يمكنك بعد ذلك تحريك (نقل) نطاق التحديد والتقاطع
# بين نطاقين لمشاهدة التقاطعات. يمكنك إلغاء تحديد عن طريق
# النقر مرة أخرى على نفس المحور.
#
# على وجه الخصوص لهذا البحث عن المعلمات، من المثير للاهتمام ملاحظة أن
# النماذج ذات الأداء الأعلى لا تعتمد على المعيار `norm`، ولكنها تعتمد على
# المقايضة بين `max_df`، و`min_df`، وقوة المعايرة `alpha`. والسبب هو أن تضمين الميزات الضجيجية (أي
# `max_df` قريب من :math:`1.0` أو `min_df` قريب من :math:`0`) يميل إلى
# الإفراط في التكيف وبالتالي يتطلب معايرة أقوى للتعويض. وجود
# ميزات أقل تتطلب معايرة أقل ووقت تسجيل أقل.
#
# يتم الحصول على أفضل درجات الدقة عندما تكون `alpha` بين :math:`10^{-6}`
# و :math:`10^0`، بغض النظر عن المعلمة `norm`.