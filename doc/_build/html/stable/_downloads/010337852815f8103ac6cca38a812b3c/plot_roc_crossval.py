"""
=============================================================
منحنى المستقبل التشغيلي (ROC) مع التحقق المتقاطع
=============================================================

يقدم هذا المثال كيفية تقدير وتصوير تباين مقياس منحنى المستقبل التشغيلي (ROC) باستخدام التحقق المتقاطع.

عادةً ما تتميز منحنيات ROC بمعدل الإيجابيات الحقيقية (TPR) على محور Y، ومعدل الإيجابيات الخاطئة (FPR) على محور X. وهذا يعني أن الركن العلوي الأيسر من الرسم البياني هو النقطة "المثالية" - حيث يكون معدل الإيجابيات الخاطئة صفرًا، ومعدل الإيجابيات الحقيقية واحدًا. وهذا ليس واقعيًا جدًا، ولكنه يعني أن المساحة الأكبر تحت المنحنى (AUC) تكون أفضل عادةً. كما أن "انحدار" منحنيات ROC مهم أيضًا، حيث أنه من المثالي تعظيم معدل الإيجابيات الحقيقية مع تقليل معدل الإيجابيات الخاطئة.

يوضح هذا المثال استجابة ROC لمجموعات بيانات مختلفة، تم إنشاؤها من التحقق المتقاطع K-fold. وبأخذ جميع هذه المنحنيات، يمكن حساب متوسط AUC، ورؤية تباين المنحنى عندما يتم تقسيم مجموعة التدريب إلى مجموعات فرعية مختلفة. وهذا يُظهر تقريبًا كيف يتأثر ناتج التصنيف بالتغيرات في بيانات التدريب، وكيف تختلف التقسيمات التي يولدها التحقق المتقاطع K-fold عن بعضها البعض.

.. note::

    راجع :ref:`sphx_glr_auto_examples_model_selection_plot_roc.py` لمكمل هذا المثال الذي يوضح استراتيجيات المتوسط لتعميم المقاييس للتصنيفات متعددة الفئات.
"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

# %%
# تحميل وإعداد البيانات
# =====================
#
# نحن نستورد :ref:`iris_dataset` الذي يحتوي على 3 فئات، كل منها
# يقابل نوعًا من نبات الزنبق. يمكن فصل فئة واحدة خطيًا
# عن الفئتين الأخريين؛ الفئتان الأخريان **ليستا** قابلتين للفصل الخطي عن بعضهما البعض.
#
# في ما يلي، نقوم بتجنيس مجموعة البيانات عن طريق إسقاط فئة "virginica"
# (`class_id=2`). وهذا يعني أن فئة "versicolor" (`class_id=1`)
# تعتبر الفئة الإيجابية و"setosa" كالفئة السلبية
# (`class_id=0`).

import numpy as np

from sklearn.datasets import load_iris

iris = load_iris()
target_names = iris.target_names
X, y = iris.data, iris.target
X, y = X[y != 2], y[y != 2]
n_samples, n_features = X.shape

# %%
# نضيف أيضًا ميزات عشوائية لجعل المشكلة أكثر صعوبة.
random_state = np.random.RandomState(0)
X = np.concatenate([X, random_state.randn(n_samples, 200 * n_features)], axis=1)

# %%
# التصنيف وتحليل ROC
# -------------------------------
#
# هنا نقوم بتشغيل مصنف :class:`~sklearn.svm.SVC` مع التحقق المتقاطع ورسم
# منحنيات ROC لكل طية. لاحظ أن الخط الأساسي لتحديد مستوى الفرصة
# (منحنى ROC المتقطع) هو مصنف سيقوم دائمًا بالتنبؤ بالفئة الأكثر تكرارًا.

import matplotlib.pyplot as plt

from sklearn import svm
from sklearn.metrics import RocCurveDisplay, auc
from sklearn.model_selection import StratifiedKFold

n_splits = 6
cv = StratifiedKFold(n_splits=n_splits)
classifier = svm.SVC(kernel="linear", probability=True, random_state=random_state)

tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)

fig, ax = plt.subplots(figsize=(6, 6))
for fold, (train, test) in enumerate(cv.split(X, y)):
    classifier.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        classifier,
        X[test],
        y[test],
        name=f"ROC fold {fold}",
        alpha=0.3,
        lw=1,
        ax=ax,
        plot_chance_level=(fold == n_splits - 1),
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)

std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)

ax.set(
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    title=f"Mean ROC curve with variability\n(Positive label '{target_names[1]}')",
)
ax.legend(loc="lower right")
plt.show()