"""
===============================
تصنيف ثنائي باستخدام AdaBoost
===============================

هذا المثال يقوم بتدريب نموذج شجرة قرار معزز باستخدام AdaBoost على مجموعة بيانات تصنيف غير خطية، مكونة من مجموعتين "Gaussian quantiles" (انظر: :func:`sklearn.datasets.make_gaussian_quantiles`) ويعرض حدود القرار ودرجات القرار. يتم عرض توزيعات درجات القرار بشكل منفصل للعينات من الفئة A والفئة B. يتم تحديد تسمية الفئة المتوقعة لكل عينة بناءً على إشارة درجة القرار. يتم تصنيف العينات التي لها درجات قرار أكبر من الصفر على أنها من الفئة B، وإلا يتم تصنيفها على أنها من الفئة A. يحدد مقدار درجة القرار درجة التشابه مع تسمية الفئة المتوقعة. بالإضافة إلى ذلك، يمكن بناء مجموعة بيانات جديدة تحتوي على نقاء مرغوب فيه من الفئة B، على سبيل المثال، عن طريق اختيار العينات فقط بدرجة قرار أعلى من قيمة معينة.
"""
# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import make_gaussian_quantiles
from sklearn.ensemble import AdaBoostClassifier
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.tree import DecisionTreeClassifier

# إنشاء مجموعة البيانات
X1, y1 = make_gaussian_quantiles(
    cov=2.0, n_samples=200, n_features=2, n_classes=2, random_state=1
)
X2, y2 = make_gaussian_quantiles(
    mean=(3, 3), cov=1.5, n_samples=300, n_features=2, n_classes=2, random_state=1
)
X = np.concatenate((X1, X2))
y = np.concatenate((y1, -y2 + 1))

# إنشاء وتدريب نموذج شجرة قرار معزز باستخدام AdaBoost
bdt = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1), n_estimators=200)
bdt.fit(X, y)

plot_colors = "br"
plot_step = 0.02
class_names = "AB"

plt.figure(figsize=(10, 5))

# رسم حدود القرار
ax = plt.subplot(121)
disp = DecisionBoundaryDisplay.from_estimator(
    bdt,
    X,
    cmap=plt.cm.Paired,
    response_method="predict",
    ax=ax,
    xlabel="x",
    ylabel="y",
)
x_min, x_max = disp.xx0.min(), disp.xx0.max()
y_min, y_max = disp.xx1.min(), disp.xx1.max()
plt.axis("tight")

# رسم نقاط التدريب
for i, n, c in zip(range(2), class_names, plot_colors):
    idx = np.where(y == i)
    plt.scatter(
        X[idx, 0],
        X[idx, 1],
        c=c,
        s=20,
        edgecolor="k",
        label="Class %s" % n,
    )
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.legend(loc="upper right")

plt.title("Decision Boundary")

# رسم درجات القرار ثنائية التصنيف
twoclass_output = bdt.decision_function(X)
plot_range = (twoclass_output.min(), twoclass_output.max())
plt.subplot(122)
for i, n, c in zip(range(2), class_names, plot_colors):
    plt.hist(
        twoclass_output[y == i],
        bins=10,
        range=plot_range,
        facecolor=c,
        label="Class %s" % n,
        alpha=0.5,
        edgecolor="k",
    )
x1, x2, y1, y2 = plt.axis()
plt.axis((x1, x2, y1, y2 * 1.2))
plt.legend(loc="upper right")
plt.ylabel("Samples")
plt.xlabel("Score")
plt.title("Decision Scores")

plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
plt.show()
