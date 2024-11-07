"""
==================================================================
الكشف عن البيانات الشاذة باستخدام عامل الانحراف المحلي (LOF)
==================================================================

خوارزمية عامل الانحراف المحلي (LOF) هي طريقة غير مشرفة للكشف عن الانحرافات
والتي تقوم بحساب انحراف الكثافة المحلية لنقطة بيانات معينة فيما يتعلق بجيرانها.
تعتبر الخوارزمية العينات التي لها كثافة أقل بكثير من جيرانها كبيانات شاذة.
يوضح هذا المثال كيفية استخدام LOF للكشف عن البيانات الشاذة.
يرجى ملاحظة أنه عند استخدام LOF للكشف عن البيانات الشاذة، يجب عدم استخدام الدوال
predict و decision_function و score_samples على مجموعة البيانات التدريبية
حيث قد يؤدي ذلك إلى نتائج خاطئة. يجب استخدام هذه الدوال فقط على البيانات الجديدة
التي لم يتم استخدامها في مجموعة التدريب، مثل X_test أو X_outliers أو meshgrid.
راجع: :ref:`User Guide <outlier_detection>`: للحصول على تفاصيل حول الفرق بين
الكشف عن الانحرافات والبيانات الشاذة، وكيفية استخدام LOF للكشف عن الانحرافات.

عدد الجيران المأخوذ في الاعتبار، (البارامتر n_neighbors) يتم تحديده عادةً 1)
بأنه أكبر من الحد الأدنى لعدد العينات التي يجب أن يحتويها التجمع، بحيث يمكن
اعتبار العينات الأخرى كبيانات شاذة محلية بالنسبة لهذا التجمع، و2) أقل من الحد
الأقصى لعدد العينات القريبة التي يمكن أن تكون بيانات شاذة محلية.
في الممارسة العملية، عادةً لا تتوفر مثل هذه المعلومات، ويبدو أن تحديد
n_neighbors=20 يعمل بشكل جيد بشكل عام.

"""

# المؤلفون: مطوري مكتبة ساي كيت ليرن
# معرف الترخيص: BSD-3-Clause

import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from sklearn.neighbors import LocalOutlierFactor

np.random.seed(42)

xx, yy = np.meshgrid(np.linspace(-5, 5, 500), np.linspace(-5, 5, 500))
# توليد ملاحظات تدريبية عادية (غير شاذة)
X = 0.3 * np.random.randn(100, 2)
X_train = np.r_[X + 2, X - 2]
# توليد ملاحظات عادية جديدة (غير شاذة)
X = 0.3 * np.random.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# توليد ملاحظات شاذة جديدة
X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))

# تدريب النموذج للكشف عن البيانات الشاذة (novelty=True)
clf = LocalOutlierFactor(n_neighbors=20, novelty=True, contamination=0.1)
clf.fit(X_train)
# لا تستخدم predict أو decision_function أو score_samples على X_train حيث
# قد يؤدي ذلك إلى نتائج خاطئة، ولكن فقط على البيانات الجديدة التي لم تستخدم
# في X_train، مثل X_test أو X_outliers أو meshgrid
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# رسم الحدود المكتسبة، والنقاط، والمتجهات الأقرب إلى المستوى
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.title("الكشف عن البيانات الشاذة باستخدام LOF")
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), 0, 7), cmap=plt.cm.PuBu)
a = plt.contour(xx, yy, Z, levels=[0], linewidths=2, colors="darkred")
plt.contourf(xx, yy, Z, levels=[0, Z.max()], colors="palevioletred")

s = 40
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")
plt.axis("tight")
plt.xlim((-5, 5))
plt.ylim((-5, 5))
plt.legend(
    [mlines.Line2D([], [], color="darkred"), b1, b2, c],
    [
        "الحدود المكتسبة",
        "الملاحظات التدريبية",
        "الملاحظات العادية الجديدة",
        "الملاحظات الشاذة الجديدة",
    ],
    loc="upper left",
    prop=matplotlib.font_manager.FontProperties(size=11),
)
plt.xlabel(
    "الأخطاء في الملاحظات العادية الجديدة: %d/40 ; الأخطاء في الملاحظات الشاذة الجديدة: %d/40"
    % (n_error_test, n_error_outliers)
)
plt.show()