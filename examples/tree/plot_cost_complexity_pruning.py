"""
========================================================
تشذيب أشجار القرار بعد التدريب باستخدام تقنية تشذيب تعقيد التكلفة
========================================================

.. currentmodule:: sklearn.tree

يوفر :class:`DecisionTreeClassifier` معاملات مثل
``min_samples_leaf`` و ``max_depth`` لمنع شجرة القرار من الإفراط في الملاءمة. وتقدم تقنية تشذيب تعقيد التكلفة خيارًا آخر للتحكم في حجم الشجرة. في
:class:`DecisionTreeClassifier`، يتم تحديد هذه التقنية بمعامل تعقيد التكلفة، ``ccp_alpha``. وتؤدي القيم الأكبر لـ ``ccp_alpha`` إلى زيادة عدد العقد التي يتم تشذيبها. هنا، نعرض فقط تأثير
``ccp_alpha`` على تنظيم الشجرة وكيفية اختيار قيمة ``ccp_alpha``
استنادًا إلى درجات التحقق.

راجع أيضًا :ref:`minimal_cost_complexity_pruning` للاطلاع على التفاصيل حول التشذيب.
"""
# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# %%
# إجمالي عدم النقاء في الأوراق مقابل قيم alpha الفعالة للشجرة المشذبة
# ---------------------------------------------------------------
# يجد تشذيب تعقيد التكلفة الأدنى بشكل متكرر العقدة ذات "أضعف
# رابط". ويتميز أضعف رابط بقيمة alpha فعالة، حيث يتم تشذيب
# العقد ذات أصغر قيمة alpha فعالة أولاً. للحصول على فكرة عن
# قيم ``ccp_alpha`` المناسبة، يوفر scikit-learn
# :func:`DecisionTreeClassifier.cost_complexity_pruning_path` الذي يعيد
# قيم alpha الفعالة وعدم النقاء الكلي للأوراق المقابل لكل خطوة في
# عملية التشذيب. مع زيادة قيمة alpha، يتم تشذيب المزيد من الشجرة، مما
# يزيد من عدم النقاء الكلي لأوراقها.
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

clf = DecisionTreeClassifier(random_state=0)
path = clf.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

# %%
# في الرسم التالي، يتم إزالة قيمة alpha الفعالة القصوى، لأنها
# تمثل الشجرة البسيطة التي تحتوي على عقدة واحدة فقط.
fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker="o", drawstyle="steps-post")
ax.set_xlabel("alpha الفعال")
ax.set_ylabel("إجمالي عدم النقاء في الأوراق")
ax.set_title("إجمالي عدم النقاء مقابل alpha الفعال لمجموعة التدريب")

# %%
# بعد ذلك، نقوم بتدريب شجرة قرار باستخدام قيم alpha الفعالة. والقيمة الأخيرة
# في ``ccp_alphas`` هي قيمة alpha التي تشذب الشجرة بالكامل،
# تاركة الشجرة، ``clfs[-1]``، بعقدة واحدة.
clfs = []
for ccp_alpha in ccp_alphas:
    clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
    clf.fit(X_train, y_train)
    clfs.append(clf)
print(
    "عدد العقد في الشجرة الأخيرة هو: {} مع ccp_alpha: {}".format(
        clfs[-1].tree_.node_count, ccp_alphas[-1]
    )
)

# %%
# بالنسبة لبقية هذا المثال، نقوم بإزالة العنصر الأخير في
# ``clfs`` و ``ccp_alphas``، لأنه يمثل الشجرة البسيطة التي تحتوي على عقدة واحدة
# فقط. هنا، نعرض أن عدد العقد وعمق الشجرة ينخفض مع زيادة قيمة alpha.
clfs = clfs[:-1]
ccp_alphas = ccp_alphas[:-1]

node_counts = [clf.tree_.node_count for clf in clfs]
depth = [clf.tree_.max_depth for clf in clfs]
fig, ax = plt.subplots(2, 1)
ax[0].plot(ccp_alphas, node_counts, marker="o", drawstyle="steps-post")
ax[0].set_xlabel("alpha")
ax[0].set_ylabel("عدد العقد")
ax[0].set_title("عدد العقد مقابل alpha")
ax[1].plot(ccp_alphas, depth, marker="o", drawstyle="steps-post")
ax[1].set_xlabel("alpha")
ax[1].set_ylabel("عمق الشجرة")
ax[1].set_title("العمق مقابل alpha")
fig.tight_layout()

# %%
# الدقة مقابل alpha لمجموعتي التدريب والاختبار
# ----------------------------------------------------
# عندما يتم تعيين ``ccp_alpha`` إلى الصفر مع الحفاظ على المعاملات الافتراضية
# الأخرى لـ :class:`DecisionTreeClassifier`، فإن الشجرة تفرط في الملاءمة، مما يؤدي إلى
# دقة تدريب 100% ودقة اختبار 88%. مع زيادة قيمة alpha، يتم تشذيب
# المزيد من الشجرة، مما ينتج شجرة قرار أكثر تعميمًا.
# في هذا المثال، يؤدي تعيين ``ccp_alpha=0.015`` إلى تعظيم دقة الاختبار.
train_scores = [clf.score(X_train, y_train) for clf in clfs]
test_scores = [clf.score(X_test, y_test) for clf in clfs]

fig, ax = plt.subplots()
ax.set_xlabel("alpha")
ax.set_ylabel("الدقة")
ax.set_title("الدقة مقابل alpha لمجموعتي التدريب والاختبار")
ax.plot(ccp_alphas, train_scores, marker="o", label="التدريب", drawstyle="steps-post")
ax.plot(ccp_alphas, test_scores, marker="o", label="الاختبار", drawstyle="steps-post")
ax.legend()
plt.show()