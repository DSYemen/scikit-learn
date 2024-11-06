"""
=======================
مثال IsolationForest
=======================

مثال يستخدم :class:`~sklearn.ensemble.IsolationForest` للكشف عن
الشذوذ.

:ref:`isolation_forest` هي مجموعة من "أشجار العزل" التي "تعزل"
الملاحظات عن طريق التقسيم العشوائي التكراري، والذي يمكن تمثيله
ببنية شجرة. يكون عدد التقسيمات المطلوبة لعزل عينة أقل
بالنسبة للقيم المتطرفة وأعلى بالنسبة للقيم الداخلية.

في هذا المثال، نعرض طريقتين لتصور حدود القرار لـ Isolation Forest المدربة على مجموعة بيانات تجريبية.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# توليد البيانات
# ---------------
#
# نقوم بإنشاء مجموعتين (كل منهما تحتوي على `n_samples`) عن طريق أخذ عينات عشوائية
# من التوزيع الطبيعي القياسي كما هو مسترجع بواسطة
# :func:`numpy.random.randn`. إحداهما كروية والأخرى
# مشوهة قليلاً.
#
# من أجل الاتساق مع تدوين :class:`~sklearn.ensemble.IsolationForest`،
# يتم تعيين تصنيف أرضي `1` للقيم الداخلية (أي المجموعات الغاوسية)
# بينما يتم تعيين التصنيف `-1` للقيم المتطرفة (التي تم إنشاؤها باستخدام :func:`numpy.random.uniform`).

import numpy as np

from sklearn.model_selection import train_test_split

n_samples, n_outliers = 120, 40
rng = np.random.RandomState(0)
covariance = np.array([[0.5, -0.1], [0.7, 0.4]])
cluster_1 = 0.4 * rng.randn(n_samples, 2) @ covariance + np.array([2, 2])  # عام
cluster_2 = 0.3 * rng.randn(n_samples, 2) + np.array([-2, -2])  # كروي
outliers = rng.uniform(low=-4, high=4, size=(n_outliers, 2))

X = np.concatenate([cluster_1, cluster_2, outliers])
y = np.concatenate(
    [np.ones((2 * n_samples), dtype=int), -np.ones((n_outliers), dtype=int)]
)

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

# %%
# يمكننا تصور المجموعات الناتجة:

import matplotlib.pyplot as plt

scatter = plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
handles, labels = scatter.legend_elements()
plt.axis("square")
plt.legend(handles=handles, labels=["القيم المتطرفة", "القيم الداخلية"], title="التصنيف الحقيقي")
plt.title("القيم الداخلية الغاوسية مع \nالقيم المتطرفة الموزعة بشكل موحد")
plt.show()

# %%
# تدريب النموذج
# ---------------------

from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples=100, random_state=0)
clf.fit(X_train)

# %%
# رسم حدود القرار المنفصلة
# -------------------------------
#
# نستخدم الفئة :class:`~sklearn.inspection.DecisionBoundaryDisplay`
# لتصور حدود القرار المنفصلة. يمثل لون الخلفية
# ما إذا كانت عينة في تلك المنطقة معينة متوقع أن تكون قيمة متطرفة
# أم لا. يعرض مخطط التشتت التصنيفات الحقيقية.

import matplotlib.pyplot as plt

from sklearn.inspection import DecisionBoundaryDisplay

disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="predict",
    alpha=0.5,
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
disp.ax_.set_title("حدود القرار الثنائية \nلـ IsolationForest")
plt.axis("square")
plt.legend(handles=handles, labels=["القيم المتطرفة", "القيم الداخلية"], title="التصنيف الحقيقي")
plt.show()

# %%
# رسم حدود قرار طول المسار
# ----------------------------------
#
# عن طريق تعيين `response_method="decision_function"`، تمثل خلفية
# :class:`~sklearn.inspection.DecisionBoundaryDisplay` مقياس
# طبيعية الملاحظة. يتم إعطاء هذه النتيجة بواسطة متوسط ​​طول المسار
# على غابة من الأشجار العشوائية، والذي يتم إعطاؤه بواسطة عمق الورقة
# (أو بشكل مكافئ عدد التقسيمات) المطلوبة لعزل عينة معينة.
#
# عندما تنتج غابة من الأشجار العشوائية بشكل جماعي أطوال مسار قصيرة
# لعزل بعض العينات المعينة، فمن المحتمل جدًا أن تكون شذوذًا
# ويكون مقياس الطبيعية قريبًا من `0`. وبشكل مشابه، تتوافق المسارات الكبيرة
# مع القيم القريبة من `1` ومن المرجح أن تكون قيمًا داخلية.

disp = DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="decision_function",
    alpha=0.5,
)
disp.ax_.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor="k")
disp.ax_.set_title("حدود قرار طول المسار \nلـ IsolationForest")
plt.axis("square")
plt.legend(handles=handles, labels=["القيم المتطرفة", "القيم الداخلية"], title="التصنيف الحقيقي")
plt.colorbar(disp.ax_.collections[1])
plt.show()


