"""
========================
إزالة الميزة المتكررة
========================

يوضح هذا المثال كيفية استخدام حذف الميزات التكراري
(:class:`~sklearn.feature_selection.RFE`) لتحديد
أهمية وحدات البكسل الفردية لتصنيف الأرقام المكتوبة بخط اليد.
:class:`~sklearn.feature_selection.RFE` يزيل بشكل تكراري الميزات الأقل
أهمية، ويخصص الرتب بناءً على أهميتها، حيث تشير قيم `ranking_` الأعلى
إلى أهمية أقل. يتم تصور الترتيب باستخدام كل من درجات اللون الأزرق
وشروح البكسل من أجل الوضوح. كما هو متوقع، تميل وحدات البكسل الموجودة
في وسط الصورة إلى أن تكون أكثر قدرة على التنبؤ من تلك القريبة من الحواف.

.. note::

    See also :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`

"""  # noqa: E501

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

# تحميل مجموعة بيانات الأرقام
digits = load_digits()
X = digits.images.reshape((len(digits.images), -1))
y = digits.target

pipe = Pipeline(
    [
        ("scaler", MinMaxScaler()),
        ("rfe", RFE(estimator=LogisticRegression(), n_features_to_select=1, step=1)),
    ]
)

pipe.fit(X, y)
ranking = pipe.named_steps["rfe"].ranking_.reshape(digits.images[0].shape)

# رسم ترتيب البكسل
plt.matshow(ranking, cmap=plt.cm.Blues)

# إضافة شروح لأرقام البكسل
for i in range(ranking.shape[0]):
    for j in range(ranking.shape[1]):
        plt.text(j, i, str(ranking[i, j]), ha="center", va="center", color="black")

plt.colorbar()
plt.title("ترتيب البكسل باستخدام RFE\n(الانحدار اللوجستي)")
plt.show()


