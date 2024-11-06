"""
=================================================
رسم تنبؤات الانحدار الفردية والتصويتية
=================================================

.. currentmodule:: sklearn

مُنحدِر التصويت هو مقدر تلوي جماعي يقوم بملاءمة العديد من المُنحدرات
الأساسية، كل منها على مجموعة البيانات بأكملها. ثم يقوم بمتوسط ​​التنبؤات
الفردية لتشكيل تنبؤ نهائي.
سنستخدم ثلاثة مُنحدرات مختلفة للتنبؤ بالبيانات:
:class:`~ensemble.GradientBoostingRegressor` و
:class:`~ensemble.RandomForestRegressor` و
:class:`~linear_model.LinearRegression`).
ثم سيتم استخدام المُنحدرات الثلاثة المذكورة أعلاه لـ
:class:`~ensemble.VotingRegressor`.

أخيرًا، سنرسم التنبؤات التي تم إجراؤها بواسطة جميع النماذج للمقارنة.

سنعمل مع مجموعة بيانات مرض السكري التي تتكون من 10 ميزات
تم جمعها من مجموعة من مرضى السكري. الهدف هو قياس كمي
لتطور المرض بعد عام واحد من خط الأساس.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

import matplotlib.pyplot as plt

from sklearn.datasets import load_diabetes
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.linear_model import LinearRegression

# %%
# تدريب المصنفات
# --------------------------------
#
# أولاً، سنقوم بتحميل مجموعة بيانات مرض السكري وبدء مُنحدِر التعزيز
# المتدرج، ومُنحدِر غابة عشوائية، وانحدار خطي. بعد ذلك، سنستخدم
# المُنحدرات الثلاثة لبناء مُنحدِر التصويت:

X, y = load_diabetes(return_X_y=True)

# تدريب المصنفات
reg1 = GradientBoostingRegressor(random_state=1)
reg2 = RandomForestRegressor(random_state=1)
reg3 = LinearRegression()

reg1.fit(X, y)
reg2.fit(X, y)
reg3.fit(X, y)

ereg = VotingRegressor([("gb", reg1), ("rf", reg2), ("lr", reg3)])
ereg.fit(X, y)

# %%
# إجراء التنبؤات
# --------------------------------
#
# الآن سنستخدم كل مُنحدِر لإجراء التنبؤات العشرين الأولى.

xt = X[:20]

pred1 = reg1.predict(xt)
pred2 = reg2.predict(xt)
pred3 = reg3.predict(xt)
pred4 = ereg.predict(xt)

# %%
# رسم النتائج
# --------------------------------
#
# أخيرًا، سنقوم بتصور التنبؤات العشرين. تُظهر النجوم الحمراء متوسط
# ​​التنبؤ الذي تم إجراؤه بواسطة :class:`~ensemble.VotingRegressor`.


plt.figure()
plt.plot(pred1, "gd", label="GradientBoostingRegressor")
plt.plot(pred2, "b^", label="RandomForestRegressor")
plt.plot(pred3, "ys", label="LinearRegression")
plt.plot(pred4, "r*", ms=10, label="VotingRegressor")

plt.tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
plt.ylabel("متوقع")
plt.xlabel("عينات التدريب")
plt.legend(loc="best")
plt.title("تنبؤات المُنحدِر ومتوسطها")

plt.show()


