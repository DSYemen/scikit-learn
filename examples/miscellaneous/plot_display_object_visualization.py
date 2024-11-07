"""
===================================
التصورات باستخدام كائنات العرض
===================================

.. currentmodule:: sklearn.metrics

في هذا المثال، سنقوم بإنشاء كائنات عرض،
:class:`ConfusionMatrixDisplay`، :class:`RocCurveDisplay`، و
:class:`PrecisionRecallDisplay` مباشرة من مقاييسها الخاصة. هذا
بديل لاستخدام وظائف الرسم الخاصة بها عندما
تكون تنبؤات النموذج محسوبة بالفعل أو مكلفة في الحساب. لاحظ أن
هذا استخدام متقدم، ونحن نوصي عمومًا باستخدام وظائف الرسم الخاصة بها.

"""

# المؤلفون: مطوري scikit-learn
# SPDX-License-Identifier: BSD-3-Clause

# %%
# تحميل البيانات وتدريب النموذج
# -------------------------
# في هذا المثال، نقوم بتحميل مجموعة بيانات مركز خدمة نقل الدم من
# `OpenML <https://www.openml.org/d/1464>`_. هذه مشكلة تصنيف ثنائي
# حيث الهدف هو ما إذا كان الفرد قد تبرع بالدم. ثم يتم تقسيم
# البيانات إلى مجموعة بيانات تدريب واختبار ويتم تثبيت الانحدار اللوجستي
# باستخدام مجموعة بيانات التدريب.
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

X, y = fetch_openml(data_id=1464, return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
clf.fit(X_train, y_train)

# %%
# إنشاء :class:`ConfusionMatrixDisplay`
# ######################################
# باستخدام النموذج المدرب، نقوم بحساب تنبؤات النموذج على مجموعة الاختبار.
# يتم استخدام هذه التنبؤات لحساب مصفوفة الارتباك التي
# يتم رسمها باستخدام :class:`ConfusionMatrixDisplay`
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

y_pred = clf.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

cm_display = ConfusionMatrixDisplay(cm).plot()


# %%
# إنشاء :class:`RocCurveDisplay`
# ###############################
# يتطلب منحنى ROC إما الاحتمالات أو قيم القرار غير المحددة
# من المقدر. نظرًا لأن الانحدار اللوجستي يوفر
# دالة قرار، فسنستخدمها لرسم منحنى ROC:
from sklearn.metrics import RocCurveDisplay, roc_curve

y_score = clf.decision_function(X_test)

fpr, tpr, _ = roc_curve(y_test, y_score, pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()

# %%
# إنشاء :class:`PrecisionRecallDisplay`
# ######################################
# وبالمثل، يمكن رسم منحنى الدقة والاستدعاء باستخدام `y_score` من
# أقسام التقدير السابقة.
from sklearn.metrics import PrecisionRecallDisplay, precision_recall_curve

prec, recall, _ = precision_recall_curve(y_test, y_score, pos_label=clf.classes_[1])
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall).plot()

# %%
# دمج كائنات العرض في رسم واحد
# ################################################
# تقوم كائنات العرض بتخزين القيم المحسوبة التي تم تمريرها كحجج.
# يسمح هذا بدمج التصورات بسهولة باستخدام واجهة برمجة التطبيقات الخاصة بـ Matplotlib.
# في المثال التالي، نقوم بوضع العروض بجانب بعضها البعض في
# صف.

# sphinx_gallery_thumbnail_number = 4
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))

roc_display.plot(ax=ax1)
pr_display.plot(ax=ax2)
plt.show()