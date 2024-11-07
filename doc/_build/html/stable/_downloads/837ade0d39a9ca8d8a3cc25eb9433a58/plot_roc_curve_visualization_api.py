"""
=============================================
منحنى ROC مع واجهة برمجة التطبيقات للتصور
=============================================
يعرّف Scikit-learn واجهة برمجة تطبيقات بسيطة لإنشاء تصورات للتعلم الآلي. الميزات الرئيسية لهذه الواجهة هي السماح بالرسم السريع والتعديلات المرئية دون إعادة الحساب. في هذا المثال، سوف نوضح كيفية استخدام واجهة برمجة التطبيقات للتصور من خلال مقارنة منحنيات ROC.

"""

# المؤلفون: مطوري scikit-learn
# معرف SPDX-License: BSD-3-Clause

# %%
# تحميل البيانات وتدريب SVC
# -------------------------
# أولاً، نقوم بتحميل مجموعة بيانات النبيذ وتحويلها إلى مشكلة تصنيف ثنائي. ثم نقوم بتدريب مصنف ناقل الدعم على مجموعة بيانات التدريب.
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

X, y = load_wine(return_X_y=True)
y = y == 2

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
svc = SVC(random_state=42)
svc.fit(X_train, y_train)

# %%
# رسم منحنى ROC
# ----------------------
# بعد ذلك، نرسم منحنى ROC باستخدام مكالمة واحدة لـ
# :func:`sklearn.metrics.RocCurveDisplay.from_estimator`. الكائن `svc_disp` الذي يتم إرجاعه يسمح لنا بالاستمرار في استخدام منحنى ROC المحسوب بالفعل
# لـ SVC في الرسوم البيانية المستقبلية.
svc_disp = RocCurveDisplay.from_estimator(svc, X_test, y_test)
plt.show()

# %%
# تدريب غابة عشوائية ورسم منحنى ROC
# ---------------------------------------------------
# نقوم بتدريب مصنف الغابة العشوائية وإنشاء رسم بياني يقارنه بمنحنى ROC لـ SVC. لاحظ كيف أن `svc_disp` يستخدم
# :func:`~sklearn.metrics.RocCurveDisplay.plot` لرسم منحنى ROC لـ SVC
# دون إعادة حساب قيم منحنى ROC نفسه. علاوة على ذلك، نقوم
# بتمرير `alpha=0.8` إلى دالات الرسم لتعديل قيم ألفا للمنحنيات.
rfc = RandomForestClassifier(n_estimators=10, random_state=42)
rfc.fit(X_train, y_train)
ax = plt.gca()
rfc_disp = RocCurveDisplay.from_estimator(rfc, X_test, y_test, ax=ax, alpha=0.8)
svc_disp.plot(ax=ax, alpha=0.8)
plt.show()