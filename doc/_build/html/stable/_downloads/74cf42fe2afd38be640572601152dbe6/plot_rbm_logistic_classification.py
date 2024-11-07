"""
==============================================================
ميزات شبكة بولتزمان المقيدة لتصنيف الأرقام
==============================================================

بالنسبة لبيانات الصور الرمادية، حيث يمكن تفسير قيم البكسل على أنها درجات من السواد على خلفية بيضاء، مثل التعرف على الأرقام المكتوبة بخط اليد، يمكن لنموذج شبكة بولتزمان المقيدة ذات التوزيع البرنولي (:class:`BernoulliRBM
<sklearn.neural_network.BernoulliRBM>`) أن يقوم باستخراج الميزات غير الخطية بشكل فعال.

"""

# المؤلفون: مطوري مكتبة سكايلرن
# معرف الترخيص: BSD-3-Clause

# %%
# توليد البيانات
# -------------
#
# من أجل تعلم تمثيلات كامنة جيدة من مجموعة بيانات صغيرة، نقوم
# بتوليد المزيد من البيانات المُعَلَّمة بشكل اصطناعي عن طريق إزعاج بيانات التدريب مع
# تحولات خطية بمقدار 1 بكسل في كل اتجاه.

import numpy as np
from scipy.ndimage import convolve

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import minmax_scale


def nudge_dataset(X, Y):
    """
    هذا ينتج مجموعة بيانات أكبر بخمس مرات من الأصلية،
    عن طريق تحريك الصور 8x8 في X حولها بمقدار 1px إلى اليسار، اليمين، الأسفل، الأعلى
    """
    direction_vectors = [
        [[0, 1, 0], [0, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 1], [0, 0, 0]],
        [[0, 0, 0], [0, 0, 0], [0, 1, 0]],
    ]

    def shift(x, w):
        return convolve(x.reshape((8, 8)), mode="constant", weights=w).ravel()

    X = np.concatenate(
        [X] + [np.apply_along_axis(shift, 1, X, vector) for vector in direction_vectors]
    )
    Y = np.concatenate([Y for _ in range(5)], axis=0)
    return X, Y


X, y = datasets.load_digits(return_X_y=True)
X = np.asarray(X, "float32")
X, Y = nudge_dataset(X, y)
X = minmax_scale(X, feature_range=(0, 1))  # 0-1 scaling

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

# %%
# تعريف النماذج
# -----------------
#
# نقوم ببناء خط أنابيب التصنيف مع مستخرج ميزات BernoulliRBM و
# مصنف :class:`LogisticRegression <sklearn.linear_model.LogisticRegression>`
#

from sklearn import linear_model
from sklearn.neural_network import BernoulliRBM
from sklearn.pipeline import Pipeline

logistic = linear_model.LogisticRegression(solver="newton-cg", tol=1)
rbm = BernoulliRBM(random_state=0, verbose=True)

rbm_features_classifier = Pipeline(steps=[("rbm", rbm), ("logistic", logistic)])

# %%
# التدريب
# --------
#
# تم تحسين فرط معاملات النموذج بالكامل (معدل التعلم، حجم الطبقة المخفية،
# التنظيم) عن طريق البحث الشبكي، ولكن البحث غير
# مستنسخ هنا بسبب قيود وقت التشغيل.

from sklearn.base import clone

# فرط المعاملات. تم ضبط هذه القيم عن طريق التحقق من الصحة المتقاطعة،
# باستخدام GridSearchCV. هنا لا نقوم بالتحقق من الصحة المتقاطعة لتوفير الوقت.
rbm.learning_rate = 0.06
rbm.n_iter = 10

# المزيد من المكونات تميل إلى إعطاء أداء تنبؤ أفضل، ولكن وقت
# ملاءمة أكبر
rbm.n_components = 100
logistic.C = 6000

# تدريب خط أنابيب RBM-Logistic
rbm_features_classifier.fit(X_train, Y_train)

# تدريب مصنف الانحدار اللوجستي مباشرة على البكسل
raw_pixel_classifier = clone(logistic)
raw_pixel_classifier.C = 100.0
raw_pixel_classifier.fit(X_train, Y_train)

# %%
# التقييم
# ----------

from sklearn import metrics

Y_pred = rbm_features_classifier.predict(X_test)
print(
    "انحدار لوجستي باستخدام ميزات RBM:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
Y_pred = raw_pixel_classifier.predict(X_test)
print(
    "انحدار لوجستي باستخدام ميزات البكسل الخام:\n%s\n"
    % (metrics.classification_report(Y_test, Y_pred))
)

# %%
# تساعد الميزات المستخرجة بواسطة BernoulliRBM في تحسين دقة التصنيف
# فيما يتعلق بالانحدار اللوجستي على البكسل الخام.

# %%
# الرسم
# --------

import matplotlib.pyplot as plt

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(rbm.components_):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape((8, 8)), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle("100 مكون مستخرج بواسطة RBM", fontsize=16)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()