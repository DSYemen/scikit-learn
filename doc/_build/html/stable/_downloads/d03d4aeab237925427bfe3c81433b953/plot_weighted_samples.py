"""
=====================
SVM: العينات ذات الأوزان
=====================

رسم دالة القرار لمجموعة بيانات ذات أوزان، حيث يتناسب حجم النقاط
مع وزنها.

يعيد التوزين عينة إعادة ضبط معامل C، مما يعني أن المصنف
يركز أكثر على الحصول على هذه النقاط بشكل صحيح. قد يكون التأثير في كثير من الأحيان دقيقًا.
لتأكيد التأثير هنا، نحن نعطي أوزانًا أكبر للبيانات الشاذة، مما يجعل
تشوه حدود القرار واضحًا جدًا.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn import svm


def plot_decision_function(classifier, sample_weight, axis, title):
    # رسم دالة القرار
    xx, yy = np.meshgrid(np.linspace(-4, 5, 500), np.linspace(-4, 5, 500))

    Z = classifier.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # رسم الخط، النقاط، وأقرب المتجهات إلى المستوى
    axis.contourf(xx, yy, Z, alpha=0.75, cmap=plt.cm.bone)
    axis.scatter(
        X[:, 0],
        X[:, 1],
        c=y,
        s=100 * sample_weight,
        alpha=0.9,
        cmap=plt.cm.bone,
        edgecolors="black",
    )

    axis.axis("off")
    axis.set_title(title)


# نقوم بإنشاء 20 نقطة
np.random.seed(0)
X = np.r_[np.random.randn(10, 2) + [1, 1], np.random.randn(10, 2)]
y = [1] * 10 + [-1] * 10
sample_weight_last_ten = abs(np.random.randn(len(X)))
sample_weight_constant = np.ones(len(X))
# وأوزان أكبر للبيانات الشاذة
sample_weight_last_ten[15:] *= 5
sample_weight_last_ten[9] *= 15

# تدريب النماذج.

# هذا النموذج لا يأخذ في الاعتبار أوزان العينات.
clf_no_weights = svm.SVC(gamma=1)
clf_no_weights.fit(X, y)

# هذا النموذج الآخر يأخذ في الاعتبار أوزان العينات المخصصة.
clf_weights = svm.SVC(gamma=1)
clf_weights.fit(X, y, sample_weight=sample_weight_last_ten)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
plot_decision_function(
    clf_no_weights, sample_weight_constant, axes[0], "أوزان ثابتة"
)
plot_decision_function(clf_weights, sample_weight_last_ten, axes[1], "أوزان معدلة")

plt.show()