"""
====================================================================
One-Class SVM مقابل One-Class SVM باستخدام Stochastic Gradient Descent
====================================================================

يوضح هذا المثال كيفية تقريب حل
:class:`sklearn.svm.OneClassSVM` في حالة استخدام نواة RBF مع
:class:`sklearn.linear_model.SGDOneClassSVM`، وهي نسخة Stochastic Gradient Descent
(SGD) من One-Class SVM. يتم استخدام تقريب النواة أولاً من أجل تطبيق
:class:`sklearn.linear_model.SGDOneClassSVM` الذي ينفذ One-Class SVM خطي باستخدام SGD.

ملاحظة: :class:`sklearn.linear_model.SGDOneClassSVM` يتناسب خطياً مع
عدد العينات في حين أن تعقيد :class:`sklearn.svm.OneClassSVM`
الذي يستخدم نواة kernelized هو على الأقل تربيعي فيما يتعلق بعدد العينات.
ليس الغرض من هذا المثال توضيح فوائد مثل هذا التقريب من حيث وقت الحساب،
ولكن بدلاً من ذلك، لإظهار أننا نحصل على نتائج مماثلة على مجموعة بيانات تجريبية.
"""
# المؤلفون: مطورو scikit-learn
# معرف الترخيص: BSD-3-Clause

# %%
import matplotlib
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDOneClassSVM
from sklearn.pipeline import make_pipeline
from sklearn.svm import OneClassSVM

line = {"weight": "normal", "size": 15}

matplotlib.rc("font", **line)

random_state = 42
rng = np.random.RandomState(random_state)

# توليد بيانات التدريب
X = 0.3 * rng.randn(500, 2)
X_train = np.r_[X + 2, X - 2]
# توليد بعض الملاحظات العادية الجديدة
X = 0.3 * rng.randn(20, 2)
X_test = np.r_[X + 2, X - 2]
# توليد بعض الملاحظات الشاذة الجديدة
X_outliers = rng.uniform(low=-4, high=4, size=(20, 2))

# فرط معلمات OCSVM
nu = 0.05
gamma = 2.0

# ملاءمة One-Class SVM
clf = OneClassSVM(gamma=gamma, kernel="rbf", nu=nu)
clf.fit(X_train)
y_pred_train = clf.predict(X_train)
y_pred_test = clf.predict(X_test)
y_pred_outliers = clf.predict(X_outliers)
n_error_train = y_pred_train[y_pred_train == -1].size
n_error_test = y_pred_test[y_pred_test == -1].size
n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

# ملاءمة One-Class SVM باستخدام تقريب النواة و SGD
transform = Nystroem(gamma=gamma, random_state=random_state)
clf_sgd = SGDOneClassSVM(
    nu=nu, shuffle=True, fit_intercept=True, random_state=random_state, tol=1e-4
)
pipe_sgd = make_pipeline(transform, clf_sgd)
pipe_sgd.fit(X_train)
y_pred_train_sgd = pipe_sgd.predict(X_train)
y_pred_test_sgd = pipe_sgd.predict(X_test)
y_pred_outliers_sgd = pipe_sgd.predict(X_outliers)
n_error_train_sgd = y_pred_train_sgd[y_pred_train_sgd == -1].size
n_error_test_sgd = y_pred_test_sgd[y_pred_test_sgd == -1].size
n_error_outliers_sgd = y_pred_outliers_sgd[y_pred_outliers_sgd == 1].size


# %%
from sklearn.inspection import DecisionBoundaryDisplay

_, ax = plt.subplots(figsize=(9, 6))

xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    cmap="PuBu",
)
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="decision_function",
    plot_method="contour",
    ax=ax,
    linewidths=2,
    colors="darkred",
    levels=[0],
)
DecisionBoundaryDisplay.from_estimator(
    clf,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    colors="palevioletred",
    levels=[0, clf.decision_function(X).max()],
)

s = 20
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")

ax.set(
    title="One-Class SVM",  # لم يتم تغيير هذا العنوان
    xlim=(-4.5, 4.5),
    ylim=(-4.5, 4.5),
    xlabel=(
        f"أخطاء التدريب: {n_error_train}/{X_train.shape[0]}; "
        f"أخطاء عادية جديدة: {n_error_test}/{X_test.shape[0]}; "
        f"أخطاء شاذة جديدة: {n_error_outliers}/{X_outliers.shape[0]}"
    ),
)
_ = ax.legend(
    [mlines.Line2D([], [], color="darkred", label="الحدود المكتسبة"), b1, b2, c],
    [
        "الحدود المكتسبة",
        "ملاحظات التدريب",
        "ملاحظات عادية جديدة",
        "ملاحظات شاذة جديدة",
    ],
    loc="upper left",
)

# %%
_, ax = plt.subplots(figsize=(9, 6))

xx, yy = np.meshgrid(np.linspace(-4.5, 4.5, 50), np.linspace(-4.5, 4.5, 50))
X = np.concatenate([xx.ravel().reshape(-1, 1), yy.ravel().reshape(-1, 1)], axis=1)
DecisionBoundaryDisplay.from_estimator(
    pipe_sgd,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    cmap="PuBu",
)
DecisionBoundaryDisplay.from_estimator(
    pipe_sgd,
    X,
    response_method="decision_function",
    plot_method="contour",
    ax=ax,
    linewidths=2,
    colors="darkred",
    levels=[0],
)
DecisionBoundaryDisplay.from_estimator(
    pipe_sgd,
    X,
    response_method="decision_function",
    plot_method="contourf",
    ax=ax,
    colors="palevioletred",
    levels=[0, pipe_sgd.decision_function(X).max()],
)

s = 20
b1 = plt.scatter(X_train[:, 0], X_train[:, 1], c="white", s=s, edgecolors="k")
b2 = plt.scatter(X_test[:, 0], X_test[:, 1], c="blueviolet", s=s, edgecolors="k")
c = plt.scatter(X_outliers[:, 0], X_outliers[:, 1], c="gold", s=s, edgecolors="k")

ax.set(
    title="One-Class SVM عبر الإنترنت",
    xlim=(-4.5, 4.5),
    ylim=(-4.5, 4.5),
    xlabel=(
        f"أخطاء التدريب: {n_error_train_sgd}/{X_train.shape[0]}; "
        f"أخطاء عادية جديدة: {n_error_test_sgd}/{X_test.shape[0]}; "
        f"أخطاء شاذة جديدة: {n_error_outliers_sgd}/{X_outliers.shape[0]}"
    ),
)
ax.legend(
    [mlines.Line2D([], [], color="darkred", label="الحدود المكتسبة"), b1, b2, c],
    [
        "الحدود المكتسبة",
        "ملاحظات التدريب",
        "ملاحظات عادية جديدة",
        "ملاحظات شاذة جديدة",
    ],
    loc="upper left",
)
plt.show()
