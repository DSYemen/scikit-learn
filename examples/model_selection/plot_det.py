"""
====================================
منحنى الخطأ الكشف (DET)
====================================

في هذا المثال، نقارن بين مقياسين متعددين للعتبات للتصنيف الثنائي:
منحنى الخاصية التشغيلية للمستقبل (ROC) ومنحنى الخطأ الكشف (DET). ولتحقيق هذا الغرض، نقوم بتقييم مصنفين مختلفين لنفس
مهمة التصنيف.

تتميز منحنيات ROC بمعدل الإيجابيات الحقيقية (TPR) على محور Y، ومعدل الإيجابيات الخاطئة (FPR) على محور X. وهذا يعني أن الركن العلوي الأيسر من الرسم البياني هو
النقطة "المثالية" - FPR صفر، و TPR واحد.

منحنيات DET هي تباين من منحنيات ROC حيث يتم رسم معدل السلبيات الخاطئة (FNR)
على محور Y بدلاً من TPR. في هذه الحالة، الأصل (الركن السفلي الأيسر) هو النقطة "المثالية".

.. note::

    - راجع :func:`sklearn.metrics.roc_curve` لمزيد من المعلومات حول منحنيات ROC.

    - راجع :func:`sklearn.metrics.det_curve` لمزيد من المعلومات حول
      منحنيات DET.

    - هذا المثال يعتمد بشكل فضفاض على
      :ref:`sphx_glr_auto_examples_classification_plot_classifier_comparison.py`
      المثال.

    - راجع :ref:`sphx_glr_auto_examples_model_selection_plot_roc_crossval.py` لمثال
      لتقدير تباين منحنيات ROC وROC-AUC.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

# %%
# توليد بيانات صناعية
# -----------------------

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X, y = make_classification(
    n_samples=1_000,
    n_features=2,
    n_redundant=0,
    n_informative=2,
    random_state=1,
    n_clusters_per_class=1,
)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)

# %%
# تحديد المصنفات
# ----------------------
#
# هنا نحدد مصنفين مختلفين. الهدف هو المقارنة البصرية لأدائهم الإحصائي عبر العتبات باستخدام منحنيات ROC وDET. لا يوجد سبب محدد لاختيار هذه المصنفات على مصنفات أخرى
# متوفرة في سكايلرن.

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.svm import LinearSVC

classifiers = {
    "Linear SVM": make_pipeline(StandardScaler(), LinearSVC(C=0.025)),
    "Random Forest": RandomForestClassifier(
        max_depth=5, n_estimators=10, max_features=1
    ),
}

# %%
# رسم منحنيات ROC وDET
# -----------------------
#
# يتم رسم منحنيات DET عادةً في مقياس الانحراف الطبيعي. لتحقيق ذلك، تحول
# عرض DET معدلات الخطأ كما هو مُعاد من قبل
# :func:`~sklearn.metrics.det_curve` ومقياس المحور باستخدام
# `scipy.stats.norm`.

import matplotlib.pyplot as plt

from sklearn.metrics import DetCurveDisplay, RocCurveDisplay

fig, [ax_roc, ax_det] = plt.subplots(1, 2, figsize=(11, 5))

for name, clf in classifiers.items():
    clf.fit(X_train, y_train)

    RocCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_roc, name=name)
    DetCurveDisplay.from_estimator(clf, X_test, y_test, ax=ax_det, name=name)

ax_roc.set_title("Receiver Operating Characteristic (ROC) curves")
ax_det.set_title("Detection Error Tradeoff (DET) curves")

ax_roc.grid(linestyle="--")
ax_det.grid(linestyle="--")

plt.legend()
plt.show()

# %%
# لاحظ أنه من الأسهل تقييم الأداء العام
# لخوارزميات التصنيف المختلفة باستخدام منحنيات DET أكثر من استخدام منحنيات ROC. نظرًا لأن منحنيات ROC يتم رسمها في مقياس خطي، عادةً ما تبدو المصنفات المختلفة
# متشابهة لجزء كبير من الرسم البياني وتختلف أكثر في الركن العلوي الأيسر
# من الرسم البياني. من ناحية أخرى، لأن منحنيات DET تمثل خطوطًا مستقيمة
# في مقياس الانحراف الطبيعي، فإنها تميل إلى أن تكون مميزة ككل
# ومنطقة الاهتمام تمتد على جزء كبير من الرسم البياني.
#
# توفر منحنيات DET تعليقات مباشرة حول خطأ الكشف التجاري لمساعدة
# في تحليل نقطة التشغيل. بعد ذلك، يمكن للمستخدم تحديد FNR الذي يرغب في
# قبوله على حساب FPR (أو العكس).