"""
===============================================================
مقارنة بين نماذج الغابات العشوائية ورفع التدرج بالرسم البياني
===============================================================

في هذا المثال، نقارن بين أداء نموذج الغابة العشوائية (RF) ونموذج رفع التدرج بالرسم البياني (HGBT) من حيث النتيجة ووقت الحساب لمجموعة بيانات الانحدار، على الرغم من أن **جميع المفاهيم المقدمة هنا تنطبق على التصنيف أيضًا**.

تتم المقارنة عن طريق تغيير المعلمات التي تتحكم في عدد الأشجار وفقًا لكل مقدر:

- `n_estimators` يتحكم في عدد الأشجار في الغابة. إنه رقم ثابت.
- `max_iter` هو العدد الأقصى للدورات في نموذج يعتمد على رفع التدرج. يتوافق عدد الدورات مع عدد الأشجار لمشاكل الانحدار والتصنيف الثنائي. علاوة على ذلك، يعتمد العدد الفعلي للأشجار التي يحتاجها النموذج على معايير التوقف.

يستخدم HGBT رفع التدرج لتحسين أداء النموذج بشكل تكراري عن طريق ملاءمة كل شجرة للانحدار السلبي لدالة الخسارة فيما يتعلق بالقيمة المتوقعة. من ناحية أخرى، تستند RFs على طريقة التجميع وتستخدم تصويت الأغلبية للتنبؤ بالنتيجة.

راجع :ref:`User Guide <ensemble>` لمزيد من المعلومات حول نماذج التجميع أو راجع :ref:`sphx_glr_auto_examples_ensemble_plot_hgbt_regression.py` لمثال يبرز بعض الميزات الأخرى لنماذج HGBT.
"""
# المؤلفون: مطوري scikit-learn
# SPDX-License-Identifier: BSD-3-Clause

# %%
# تحميل مجموعة البيانات
# ------------

from plotly.subplots import make_subplots
import plotly.express as px
import plotly.colors as colors
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
import pandas as pd
import joblib
from sklearn.datasets import fetch_california_housing

X, y = fetch_california_housing(return_X_y=True, as_frame=True)
n_samples, n_features = X.shape

# %%
# يستخدم HGBT خوارزمية تعتمد على الرسم البياني لقيم الميزات التي يمكنها
# التعامل بكفاءة مع مجموعات البيانات الكبيرة (عشرات الآلاف من العينات أو أكثر) مع
# عدد كبير من الميزات (انظر :ref:`Why_it's_faster`). لا يستخدم تنفيذ scikit-learn لـ RF التجميع ويعتمد على التقسيم الدقيق، والذي
# يمكن أن يكون مكلفًا من الناحية الحسابية.

print(f"تتكون مجموعة البيانات من {n_samples} عينات و {n_features} ميزات")

# %%
# حساب النتيجة وأوقات الحساب
# -----------------------------------
#
# لاحظ أن العديد من أجزاء تنفيذ
# :class:`~sklearn.ensemble.HistGradientBoostingClassifier` و
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` موازية بشكل افتراضي.
#
# يمكن أيضًا تشغيل تنفيذ :class:`~sklearn.ensemble.RandomForestRegressor` و
# :class:`~sklearn.ensemble.RandomForestClassifier` على عدة
# أنوية باستخدام معلمة `n_jobs`، هنا تم تعيينها لمطابقة عدد
# الأنوية المادية على الجهاز المضيف. راجع :ref:`parallelism` لمزيد من
# المعلومات.


N_CORES = joblib.cpu_count(only_physical_cores=True)
print(f"عدد الأنوية المادية: {N_CORES}")

# %%
# على عكس RF، توفر نماذج HGBT خيار التوقف المبكر (انظر
# :ref:`sphx_glr_auto_examples_ensemble_plot_gradient_boosting_early_stopping.py`)
# لتجنب إضافة أشجار غير ضرورية. داخليًا، يستخدم الخوارزمية مجموعة خارج العينة لحساب أداء تعميم النموذج
# عند كل إضافة لشجرة. وبالتالي، إذا لم يتحسن أداء التعميم لأكثر من `n_iter_no_change` دورات، فإنه يتوقف عن إضافة الأشجار.
#
# تم ضبط المعلمات الأخرى لكلا النموذجين ولكن الإجراء غير موضح
# هنا للحفاظ على بساطة المثال.


models = {
    "Random Forest": RandomForestRegressor(
        min_samples_leaf=5, random_state=0, n_jobs=N_CORES
    ),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_leaf_nodes=15, random_state=0, early_stopping=False
    ),
}
param_grids = {
    "Random Forest": {"n_estimators": [10, 20, 50, 100]},
    "Hist Gradient Boosting": {"max_iter": [10, 20, 50, 100, 300, 500]},
}
cv = KFold(n_splits=4, shuffle=True, random_state=0)
results = []
for name, model in models.items():
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grids[name],
        return_train_score=True,
        cv=cv,
    ).fit(X, y)
    result = {"model": name, "cv_results": pd.DataFrame(
        grid_search.cv_results_)}
    results.append(result)

# %%
# .. Note::
#  ضبط `n_estimators` لـ RF يؤدي عادة إلى إهدار طاقة الكمبيوتر. في الممارسة العملية، يحتاج المرء فقط إلى التأكد من أنه كبير بما يكفي بحيث
#  لا يؤدي مضاعفة قيمته إلى تحسين كبير لنتيجة الاختبار.
#
# رسم النتائج
# ------------
# يمكننا استخدام `plotly.express.scatter
# <https://plotly.com/python-api-reference/generated/plotly.express.scatter.html>`_
# لتصور المقايضة بين وقت الحساب المنقضي ومتوسط نتيجة الاختبار.
# تمرير المؤشر فوق نقطة معينة يعرض المعلمات المقابلة.
# أشرطة الخطأ تقابل انحرافًا معياريًا واحدًا كما هو محدد في الطيات المختلفة
# للتحقق المتقاطع.


fig = make_subplots(
    rows=1,
    cols=2,
    shared_yaxes=True,
    subplot_titles=["Train time vs score", "Predict time vs score"],
)
model_names = [result["model"] for result in results]
colors_list = colors.qualitative.Plotly * (
    len(model_names) // len(colors.qualitative.Plotly) + 1
)

for idx, result in enumerate(results):
    cv_results = result["cv_results"].round(3)
    model_name = result["model"]
    param_name = list(param_grids[model_name].keys())[0]
    cv_results[param_name] = cv_results["param_" + param_name]
    cv_results["model"] = model_name

    scatter_fig = px.scatter(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
        error_x="std_fit_time",
        error_y="std_test_score",
        hover_data=param_name,
        color="model",
    )
    line_fig = px.line(
        cv_results,
        x="mean_fit_time",
        y="mean_test_score",
    )

    scatter_trace = scatter_fig["data"][0]
    line_trace = line_fig["data"][0]
    scatter_trace.update(marker=dict(color=colors_list[idx]))
    line_trace.update(line=dict(color=colors_list[idx]))
    fig.add_trace(scatter_trace, row=1, col=1)
    fig.add_trace(line_trace, row=1, col=1)

    scatter_fig = px.scatter(
        cv_results,
        x="mean_score_time",
        y="mean_test_score",
        error_x="std_score_time",
        error_y="std_test_score",
        hover_data=param_name,
    )
    line_fig = px.line(
        cv_results,
        x="mean_score_time",
        y="mean_test_score",
    )

    scatter_trace = scatter_fig["data"][0]
    line_trace = line_fig["data"][0]
    scatter_trace.update(marker=dict(color=colors_list[idx]))
    line_trace.update(line=dict(color=colors_list[idx]))
    fig.add_trace(scatter_trace, row=1, col=2)
    fig.add_trace(line_trace, row=1, col=2)

fig.update_layout(
    xaxis=dict(title="Train time (s) - lower is better"),
    yaxis=dict(title="Test R2 score - higher is better"),
    xaxis2=dict(title="Predict time (s) - lower is better"),
    legend=dict(x=0.72, y=0.05, traceorder="normal", borderwidth=1),
    title=dict(x=0.5, text="Speed-score trade-off of tree-based ensembles"),
)
# %%
# يتحسن كل من نماذج HGBT وRF عند زيادة عدد الأشجار في
# التجميع. ومع ذلك، تصل النتائج إلى مستوى ثابت حيث يؤدي إضافة أشجار جديدة فقط
# إلى جعل الملاءمة والتسجيل أبطأ. يصل نموذج RF إلى هذا المستوى الثابت في وقت سابق
# ولا يمكنه أبدًا الوصول إلى نتيجة اختبار أكبر نموذج HGBDT.
#
# لاحظ أن النتائج المعروضة في الرسم البياني أعلاه يمكن أن تتغير قليلاً عبر الجولات
# وحتى بشكل أكبر عند تشغيلها على أجهزة أخرى: حاول تشغيل هذا
# المثال على جهازك المحلي.
#
# بشكل عام، يجب أن يلاحظ المرء غالبًا أن نماذج رفع التدرج القائمة على الرسم البياني
# تهيمن بشكل موحد على نماذج الغابات العشوائية في "نتيجة الاختبار مقابل
# مقايضة سرعة التدريب" (يجب أن يكون منحنى HGBDT في أعلى يسار منحنى RF، دون
# أن يتقاطع أبدًا). يمكن أيضًا أن تكون مقايضة "نتيجة الاختبار مقابل سرعة التنبؤ"
# أكثر تنازعًا، ولكنها في معظم الأحيان مواتية لـ HGBDT. من الجيد دائمًا التحقق من كلا النوعين من النماذج
# (مع ضبط فرط المعلمات) ومقارنة أدائها على مشكلتك المحددة لتحديد النموذج الذي
# أفضل ملاءمة ولكن **HGBT توفر دائمًا مقايضة سرعة-دقة أكثر ملاءمة من RF**، إما مع فرط المعلمات الافتراضية أو بما في ذلك
# تكلفة ضبط فرط المعلمات.
#
# هناك استثناء واحد لهذه القاعدة العامة على الرغم من ذلك: عند تدريب
# نموذج تصنيف متعدد الفئات مع عدد كبير من الفئات المحتملة، يلائم HGBDT داخليًا شجرة واحدة لكل فئة في كل دورة رفع التدرج بينما الأشجار
# التي تستخدمها نماذج RF متعددة الفئات بشكل طبيعي والتي يجب أن تحسن مقايضة السرعة والدقة
# من نماذج RF في هذه الحالة.
