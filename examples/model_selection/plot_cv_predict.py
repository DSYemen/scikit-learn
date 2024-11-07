"""
====================================
رسم تنبؤات الصندوق الأسود
====================================

هذا المثال يوضح كيفية استخدام
:func:`~sklearn.model_selection.cross_val_predict` مع
:class:`~sklearn.metrics.PredictionErrorDisplay` لتصور أخطاء التنبؤ.
"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

# %%
# سنقوم بتحميل مجموعة بيانات مرض السكري وإنشاء مثيل لنموذج الانحدار الخطي.
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

X, y = load_diabetes(return_X_y=True)
lr = LinearRegression()

# %%
# :func:`~sklearn.model_selection.cross_val_predict` يعيد مصفوفة بنفس حجم `y` حيث كل إدخال هو تنبؤ تم الحصول عليه من خلال الصندوق الأسود.
from sklearn.model_selection import cross_val_predict

y_pred = cross_val_predict(lr, X, y, cv=10)

# %%
# بما أن `cv=10`، فهذا يعني أننا قمنا بتدريب 10 نماذج وتم استخدام كل نموذج للتنبؤ على واحدة من الطيات العشر. يمكننا الآن استخدام
# :class:`~sklearn.metrics.PredictionErrorDisplay` لتصور أخطاء التنبؤ.

# على المحور الأيسر، نرسم القيم الملاحظة :math:`y` مقابل القيم المتوقعة
# :math:`\hat{y}` التي تعطيها النماذج. على المحور الأيمن، نرسم
# المتبقيات (أي الفرق بين القيم الملاحظة والقيم المتوقعة) مقابل القيم المتوقعة.
import matplotlib.pyplot as plt

from sklearn.metrics import PredictionErrorDisplay

fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
PredictionErrorDisplay.from_predictions(
    y,
    y_pred=y_pred,
    kind="actual_vs_predicted",
    subsample=100,
    ax=axs[0],
    random_state=0,
)
axs[0].set_title("القيم الفعلية مقابل القيم المتوقعة")
PredictionErrorDisplay.from_predictions(
    y,
    y_pred=y_pred,
    kind="residual_vs_predicted",
    subsample=100,
    ax=axs[1],
    random_state=0,
)
axs[1].set_title("المتبقيات مقابل القيم المتوقعة")
fig.suptitle("رسم تنبؤات الصندوق الأسود")
plt.tight_layout()
plt.show()

# %%
# من المهم ملاحظة أننا استخدمنا
# :func:`~sklearn.model_selection.cross_val_predict` لأغراض العرض فقط في هذا المثال.

# سيكون من المشكلات تقييم أداء النموذج بشكل كمي من خلال حساب مقياس أداء واحد من التنبؤات المجمعة التي تم إرجاعها بواسطة
# :func:`~sklearn.model_selection.cross_val_predict`
# عندما تختلف الطيات المختلفة للصندوق الأسود في الحجم والتوزيعات.

# يوصى بحساب مقاييس أداء لكل طية باستخدام:
# :func:`~sklearn.model_selection.cross_val_score` أو
# :func:`~sklearn.model_selection.cross_validate` بدلاً من ذلك.