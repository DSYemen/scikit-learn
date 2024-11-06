"""
===================================================
إزالة الميزات المتكررة باستخدام التحقق المتبادل
===================================================

مثال على حذف الميزات التكراري (RFE) مع الضبط التلقائي لعدد
الميزات المحددة مع التحقق المتبادل.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# توليد البيانات
# ---------------
#
# نقوم ببناء مهمة تصنيف باستخدام 3 ميزات إعلامية. إن إدخال
# ميزتين إضافيتين متكررتين (أي مترابطتين) له تأثير أن الميزات
# المحددة تختلف اعتمادًا على طية التحقق المتبادل. الميزات المتبقية
# غير إعلامية حيث يتم رسمها عشوائيًا.

from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=500,
    n_features=15,
    n_informative=3,
    n_redundant=2,
    n_repeated=0,
    n_classes=8,
    n_clusters_per_class=1,
    class_sep=0.8,
    random_state=0,
)

# %%
# تدريب النموذج واختياره
# ----------------------------
#
# نقوم بإنشاء كائن RFE وحساب الدرجات التي تم التحقق منها بشكل متبادل.
# استراتيجية التسجيل "الدقة" تعمل على تحسين نسبة العينات المصنفة بشكل صحيح.

from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

min_features_to_select = 1  # الحد الأدنى لعدد الميزات المطلوب مراعاتها
clf = LogisticRegression()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(X, y)

print(f"العدد الأمثل للميزات: {rfecv.n_features_}")

# %%
# في الحالة الحالية، تم العثور على النموذج الذي يحتوي على 3 ميزات (والذي يتوافق مع
# نموذج التوليد الحقيقي) هو الأمثل.
#
# رسم عدد الميزات مقابل درجات التحقق المتبادل
# ---------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd

cv_results = pd.DataFrame(rfecv.cv_results_)
plt.figure()
plt.xlabel("عدد الميزات المحددة")
plt.ylabel("متوسط دقة الاختبار")
plt.errorbar(
    x=cv_results["n_features"],
    y=cv_results["mean_test_score"],
    yerr=cv_results["std_test_score"],
)
plt.title("حذف الميزات التكراري \nمع ميزات مترابطة")
plt.show()

# %%
# من الرسم البياني أعلاه، يمكن للمرء أن يلاحظ أيضًا هضبة من الدرجات المتكافئة
# (متوسط قيمة متشابه وأشرطة خطأ متداخلة) لـ 3 إلى 5 ميزات محددة.
# هذه هي نتيجة إدخال ميزات مترابطة. في الواقع، يمكن أن يقع النموذج
# الأمثل الذي تم اختياره بواسطة RFE ضمن هذا النطاق، اعتمادًا على تقنية
# التحقق المتبادل. تنخفض دقة الاختبار فوق 5 ميزات محددة، وهذا يعني أن
# الاحتفاظ بالميزات غير الإعلامية يؤدي إلى فرط التخصيص وبالتالي فهو
# ضار بالأداء الإحصائي للنماذج.


