"""
===========================================================
رسم احتمالات الفئات المحسوبة بواسطة VotingClassifier
===========================================================

.. currentmodule:: sklearn

رسم احتمالات الفئات للعينة الأولى في مجموعة بيانات تجريبية متوقعة بواسطة
ثلاثة مصنفات مختلفة ومتوسط بواسطة
:class:`~ensemble.VotingClassifier`.

أولاً، تتم تهيئة ثلاثة مصنفات نموذجية
(:class:`~linear_model.LogisticRegression` و :class:`~naive_bayes.GaussianNB`
و :class:`~ensemble.RandomForestClassifier`) وتستخدم لتهيئة
:class:`~ensemble.VotingClassifier` للتصويت الناعم مع أوزان `[1، 1، 5]`، مما
يعني أن احتمالات التنبؤ لـ
:class:`~ensemble.RandomForestClassifier` تحسب 5 مرات بقدر أوزان
المصنفات الأخرى عند حساب الاحتمال المتوسط.

لتصور ترجيح الاحتمال، نقوم بملاءمة كل مصنف على مجموعة التدريب
ورسم احتمالات الفئات المتوقعة للعينة الأولى في مجموعة البيانات
النموذجية هذه.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

clf1 = LogisticRegression(max_iter=1000, random_state=123)
clf2 = RandomForestClassifier(n_estimators=100, random_state=123)
clf3 = GaussianNB()
X = np.array([[-1.0, -1.0], [-1.2, -1.4], [-3.4, -2.2], [1.1, 1.2]])
y = np.array([1, 1, 2, 2])

eclf = VotingClassifier(
    estimators=[("lr", clf1), ("rf", clf2), ("gnb", clf3)],
    voting="soft",
    weights=[1, 1, 5],
)

# تنبؤ باحتمالات الفئات لجميع المصنفات
probas = [c.fit(X, y).predict_proba(X) for c in (clf1, clf2, clf3, eclf)]

# الحصول على احتمالات الفئات للعينة الأولى في مجموعة البيانات
class1_1 = [pr[0, 0] for pr in probas]
class2_1 = [pr[0, 1] for pr in probas]


# الرسم

N = 4  # عدد المجموعات
ind = np.arange(N)  # مواضع المجموعات
width = 0.35  # عرض الشريط

fig, ax = plt.subplots()

# أشرطة للمصنف 1-3
p1 = ax.bar(ind, np.hstack(([class1_1[:-1], [0]])), width, color="green", edgecolor="k")
p2 = ax.bar(
    ind + width,
    np.hstack(([class2_1[:-1], [0]])),
    width,
    color="lightgreen",
    edgecolor="k",
)

# أشرطة لـ VotingClassifier
p3 = ax.bar(ind, [0, 0, 0, class1_1[-1]], width, color="blue", edgecolor="k")
p4 = ax.bar(
    ind + width, [0, 0, 0, class2_1[-1]], width, color="steelblue", edgecolor="k"
)

# تعليقات بيانية
plt.axvline(2.8, color="k", linestyle="dashed")
ax.set_xticks(ind + width)
ax.set_xticklabels(
    [
        "LogisticRegression\n الوزن 1",
        "GaussianNB\n الوزن 1",
        "RandomForestClassifier\n الوزن 5",
        "VotingClassifier\n (متوسط الاحتمالات)",
    ],
    rotation=40,
    ha="right",
)
plt.ylim([0, 1])
plt.title("احتمالات الفئات للعينة 1 بواسطة مصنفات مختلفة")
plt.legend([p1[0], p2[0]], ["الفئة 1", "الفئة 2"], loc="upper left")
plt.tight_layout()
plt.show()
