"""
============================================
الرسم المتقدم باستخدام الاعتمادية الجزئية
============================================
يمكن استخدام كائن :class:`~sklearn.inspection.PartialDependenceDisplay` للرسم دون الحاجة إلى إعادة حساب الاعتمادية الجزئية. في هذا المثال، نوضح كيفية رسم مخططات الاعتمادية الجزئية وكيفية تخصيص المخطط بسرعة باستخدام واجهة برمجة التطبيقات (API) للتصور.

.. note::

    راجع أيضًا :ref:`sphx_glr_auto_examples_miscellaneous_plot_roc_curve_visualization_api.py`
"""

# المؤلفون: مطوري scikit-learn
# معرف SPDX-License: BSD-3-Clause

import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.inspection import PartialDependenceDisplay
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

# %%
# تدريب النماذج على مجموعة بيانات مرض السكري
# ================================================
#
# أولاً، نقوم بتدريب شجرة قرار وشبكة عصبية متعددة الطبقات على مجموعة بيانات مرض السكري.

diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

tree = DecisionTreeRegressor()
mlp = make_pipeline(
    StandardScaler(),
    MLPRegressor(hidden_layer_sizes=(100, 100), tol=1e-2, max_iter=500, random_state=0),
)
tree.fit(X, y)
mlp.fit(X, y)

# %%
# رسم الاعتمادية الجزئية لميزتين
# ============================================
#
# نرسم منحنيات الاعتمادية الجزئية للميزتين "العمر" و"مؤشر كتلة الجسم" لشجرة القرار. مع ميزتين،
# :func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` يتوقع رسم
# منحنيين. هنا تقوم دالة الرسم بوضع شبكة من مخططين باستخدام المساحة
# المحددة بواسطة `ax` .
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Decision Tree")
tree_disp = PartialDependenceDisplay.from_estimator(tree, X, ["age", "bmi"], ax=ax)

# %%
# يمكن رسم منحنيات الاعتمادية الجزئية للشبكة العصبية متعددة الطبقات.
# في هذه الحالة، يتم تمرير `line_kw` إلى
# :func:`~sklearn.inspection.PartialDependenceDisplay.from_estimator` لتغيير
# لون المنحنى.
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_title("Multi-layer Perceptron")
mlp_disp = PartialDependenceDisplay.from_estimator(
    mlp, X, ["age", "bmi"], ax=ax, line_kw={"color": "red"}
)

# %%
# رسم الاعتمادية الجزئية للنموذجين معًا
# ======================================================
#
# تحتوي كائنات `tree_disp` و`mlp_disp`
# :class:`~sklearn.inspection.PartialDependenceDisplay` على جميع المعلومات المحسوبة اللازمة لإعادة إنشاء منحنيات الاعتمادية الجزئية. هذا
# يعني أنه يمكننا بسهولة إنشاء مخططات إضافية دون الحاجة إلى إعادة حساب
# المنحنيات.
#
# إحدى طرق رسم المنحنيات هي وضعها في نفس الشكل، مع
# منحنيات كل نموذج في كل صف. أولاً، نقوم بإنشاء شكل مع محورين
# داخل صفين وعمود واحد. يتم تمرير المحورين إلى
# :func:`~sklearn.inspection.PartialDependenceDisplay.plot` وظائف `tree_disp` و`mlp_disp`. سيتم استخدام المحاور المعطاة بواسطة دالة الرسم لرسم الاعتمادية الجزئية. يضع المخطط الناتج منحنيات الاعتمادية الجزئية لشجرة القرار في الصف الأول من
# الشبكة العصبية متعددة الطبقات في الصف الثاني.

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
tree_disp.plot(ax=ax1)
ax1.set_title("Decision Tree")
mlp_disp.plot(ax=ax2, line_kw={"color": "red"})
ax2.set_title("Multi-layer Perceptron")

# %%
# طريقة أخرى لمقارنة المنحنيات هي رسمها فوق بعضها البعض. هنا،
# نقوم بإنشاء شكل مع صف واحد وعمودين. يتم تمرير المحاور إلى
# :func:`~sklearn.inspection.PartialDependenceDisplay.plot` الدالة كقائمة،
# والتي سترسم منحنيات الاعتمادية الجزئية لكل نموذج على نفس المحاور.
# يجب أن يكون طول قائمة المحاور مساويًا لعدد المخططات المرسومة.

# sphinx_gallery_thumbnail_number = 4
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
tree_disp.plot(ax=[ax1, ax2], line_kw={"label": "Decision Tree"})
mlp_disp.plot(
    ax=[ax1, ax2], line_kw={"label": "Multi-layer Perceptron", "color": "red"}
)
ax1.legend()
ax2.legend()

# %%
# `tree_disp.axes_` هو حاوية مصفوفة numpy للمحاور المستخدمة لرسم
# مخططات الاعتمادية الجزئية. يمكن تمرير هذا إلى `mlp_disp` للحصول على نفس
# تأثير رسم المخططات فوق بعضها البعض. علاوة على ذلك، فإن
# `mlp_disp.figure_` يحفظ الشكل، مما يسمح بتغيير حجم الشكل
# بعد استدعاء `plot`. في هذه الحالة، يكون لـ `tree_disp.axes_` بعدين، وبالتالي
# لن تعرض الدالة `plot` سوى تسمية المحور y وعلامات المحور y على المخطط الأيسر.

tree_disp.plot(line_kw={"label": "Decision Tree"})
mlp_disp.plot(
    line_kw={"label": "Multi-layer Perceptron", "color": "red"}, ax=tree_disp.axes_
)
tree_disp.figure_.set_size_inches(10, 6)
tree_disp.axes_[0, 0].legend()
tree_disp.axes_[0, 1].legend()
plt.show()

# %%
# رسم الاعتمادية الجزئية لميزة واحدة
# ===========================================
#
# هنا، نرسم منحنيات الاعتمادية الجزئية لميزة واحدة، "العمر"، على
# نفس المحاور. في هذه الحالة، يتم تمرير `tree_disp.axes_` إلى دالة الرسم الثانية.
tree_disp = PartialDependenceDisplay.from_estimator(tree, X, ["age"])
mlp_disp = PartialDependenceDisplay.from_estimator(
    mlp, X, ["age"], ax=tree_disp.axes_, line_kw={"color": "red"}
)