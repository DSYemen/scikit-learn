"""
=========================================
التصنيف المتداخل مقابل غير المتداخل
=========================================

يقارن هذا المثال بين استراتيجيات التصنيف غير المتداخل والمتداخل على مصنف لمجموعة بيانات زهرة الزنبق. غالبًا ما يتم استخدام التصنيف المتداخل (CV) لتدريب نموذج تحتاج فيه أيضًا إلى تحسين المعلمات الفائقة. ويقدر التصنيف المتداخل خطأ التعميم للنموذج الأساسي و(بحث المعلمات) الفائقة. يؤدي اختيار المعلمات التي تعظم التصنيف غير المتداخل إلى تحيز النموذج إلى مجموعة البيانات، مما يؤدي إلى نتيجة متفائلة للغاية.

يستخدم اختيار النموذج بدون التصنيف المتداخل نفس البيانات لضبط معلمات النموذج وتقييم أداء النموذج. وبالتالي، قد "يتسرب" المعلومات إلى النموذج ويبالغ في ملاءمة البيانات. يعتمد حجم هذا التأثير بشكل أساسي على حجم مجموعة البيانات واستقرار النموذج. راجع Cawley و Talbot [1]_ لتحليل هذه القضايا.

لتجنب هذه المشكلة، يستخدم التصنيف المتداخل بشكل فعال سلسلة من مجموعات التدريب/التحقق/الاختبار. في الحلقة الداخلية (هنا يتم تنفيذها بواسطة :class:`GridSearchCV <sklearn.model_selection.GridSearchCV>`)، يتم تعظيم النتيجة تقريبًا عن طريق ملاءمة نموذج لكل مجموعة تدريب، ثم تعظيمها مباشرة في اختيار المعلمات الفائقة على مجموعة التحقق. في الحلقة الخارجية (هنا في :func:`cross_val_score <sklearn.model_selection.cross_val_score>`)، يتم تقدير خطأ التعميم عن طريق حساب متوسط نتائج مجموعة الاختبار على عدة تقسيمات لمجموعة البيانات.

يستخدم المثال أدناه مصنفًا متجهًا داعمًا بلب غير خطي لبناء نموذج بمعلمات فائقة محسنة عن طريق البحث الشبكي. نقارن أداء استراتيجيات التصنيف غير المتداخل والمتداخل عن طريق حساب الفرق بين نتائجها.

.. seealso::

    - :ref:`cross_validation`
    - :ref:`grid_search`

.. rubric:: المراجع

.. [1] `Cawley, G.C.; Talbot, N.L.C. On over-fitting in model selection and
    subsequent selection bias in performance evaluation.
    J. Mach. Learn. Res 2010,11, 2079-2107.
    <http://jmlr.csail.mit.edu/papers/volume11/cawley10a/cawley10a.pdf>`_
"""
# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import numpy as np
from matplotlib import pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn.svm import SVC

# عدد المحاولات العشوائية
NUM_TRIALS = 30

# تحميل مجموعة البيانات
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# تحديد القيم المحتملة للمعلمات التي سيتم تحسينها
p_grid = {"C": [1, 10, 100], "gamma": [0.01, 0.1]}

# سنستخدم مصنف متجه داعم بلب "rbf"
svm = SVC(kernel="rbf")

# مصفوفات لحفظ النتائج
non_nested_scores = np.zeros(NUM_TRIALS)
nested_scores = np.zeros(NUM_TRIALS)

# حلقة تكرارية لكل محاولة
for i in range(NUM_TRIALS):
    # اختيار تقنيات التصنيف المتداخل وغير المتداخل للحلقتين الداخلية والخارجية،
    # بشكل مستقل عن مجموعة البيانات.
    # على سبيل المثال "GroupKFold"، "LeaveOneOut"، "LeaveOneGroupOut"، إلخ.
    inner_cv = KFold(n_splits=4, shuffle=True, random_state=i)
    outer_cv = KFold(n_splits=4, shuffle=True, random_state=i)

    # البحث عن المعلمات وتسجيل النتائج بدون تداخل
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=outer_cv)
    clf.fit(X_iris, y_iris)
    non_nested_scores[i] = clf.best_score_

    # التصنيف المتداخل مع تحسين المعلمات
    clf = GridSearchCV(estimator=svm, param_grid=p_grid, cv=inner_cv)
    nested_score = cross_val_score(clf, X=X_iris, y=y_iris, cv=outer_cv)
    nested_scores[i] = nested_score.mean()

score_difference = non_nested_scores - nested_scores

print(
    "متوسط الفرق {:6f} مع انحراف معياري {:6f}.".format(
        score_difference.mean(), score_difference.std()
    )
)

# رسم النتائج لكل محاولة للتصنيف المتداخل وغير المتداخل
plt.figure()
plt.subplot(211)
(non_nested_scores_line,) = plt.plot(non_nested_scores, color="r")
(nested_line,) = plt.plot(nested_scores, color="b")
plt.ylabel("النتيجة", fontsize="14")
plt.legend(
    [non_nested_scores_line, nested_line],
    ["التصنيف غير المتداخل", "التصنيف المتداخل"],
    bbox_to_anchor=(0, 0.4, 0.5, 0),
)
plt.title(
    "التصنيف غير المتداخل والمتداخل على مجموعة بيانات زهرة الزنبق",
    x=0.5,
    y=1.1,
    fontsize="15",
)

# رسم مخطط شريطي للفرق
plt.subplot(212)
difference_plot = plt.bar(range(NUM_TRIALS), score_difference)
plt.xlabel("رقم المحاولة الفردية")
plt.legend(
    [difference_plot],
    ["التصنيف غير المتداخل - نتيجة التصنيف المتداخل"],
    bbox_to_anchor=(0, 1, 0.8, 0),
)
plt.ylabel("فرق النتيجة", fontsize="14")

plt.show()