"""
==========================================================================
توضيح العملية الغاوسية المسبقة واللاحقة لنوى مختلفة
==========================================================================

يوضح هذا المثال التوزيع المسبق واللاحق لـ :class:`~sklearn.gaussian_process.GaussianProcessRegressor` مع نوى مختلفة. يتم عرض المتوسط والانحراف المعياري و 5 عينات لكل من التوزيعات المسبقة واللاحقة.

هنا، نعطي فقط بعض الرسوم التوضيحية. لمعرفة المزيد عن صياغة النوى، ارجع إلى :ref:`دليل المستخدم <gp_kernels>`.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# دالة مساعدة
# ---------------
#
# قبل تقديم كل نواة متاحة بشكل فردي للعمليات الغاوسية، سنحدد دالة مساعدة تسمح لنا برسم عينات مأخوذة من العملية الغاوسية.
#
# ستأخذ هذه الدالة نموذج :class:`~sklearn.gaussian_process.GaussianProcessRegressor` وستقوم برسم عينات من العملية الغاوسية. إذا لم يتم ملاءمة النموذج، فسيتم رسم العينات من التوزيع المسبق، بينما بعد ملاءمة النموذج، فسيتم رسم العينات من التوزيع اللاحق.
import matplotlib.pyplot as plt
import numpy as np


def plot_gpr_samples(gpr_model, n_samples, ax):
    """ارسم عينات مأخوذة من نموذج العملية الغاوسية.

    إذا لم يتم تدريب نموذج العملية الغاوسية، فسيتم رسم العينات المأخوذة من التوزيع المسبق. خلاف ذلك، يتم رسم العينات من التوزيع اللاحق. انتبه إلى أن العينة هنا تتوافق مع دالة.

    المعلمات
    ----------
    gpr_model : `GaussianProcessRegressor`
        نموذج :class:`~sklearn.gaussian_process.GaussianProcessRegressor`.
    n_samples : int
        عدد العينات المراد رسمها من توزيع العملية الغاوسية.
    ax : محور matplotlib
        محور matplotlib حيث يتم رسم العينات.
    """
    x = np.linspace(0, 5, 100)
    X = x.reshape(-1, 1)

    y_mean, y_std = gpr_model.predict(X, return_std=True)
    y_samples = gpr_model.sample_y(X, n_samples)

    for idx, single_prior in enumerate(y_samples.T):
        ax.plot(
            x,
            single_prior,
            linestyle="--",
            alpha=0.7,
            label=f"الدالة المعينة #{idx + 1}",
        )
    ax.plot(x, y_mean, color="black", label="المتوسط")
    ax.fill_between(
        x,
        y_mean - y_std,
        y_mean + y_std,
        alpha=0.1,
        color="black",
        label=r"$\pm$ 1 الانحراف المعياري",
    )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_ylim([-3, 3])


# %%
# توليد مجموعة البيانات والعملية الغاوسية
# ---------------------------------------
# سننشئ مجموعة بيانات تدريب سنستخدمها في الأقسام المختلفة.
rng = np.random.RandomState(4)
X_train = rng.uniform(0, 5, 10).reshape(-1, 1)
y_train = np.sin((X_train[:, 0] - 2.5) ** 2)
n_samples = 5

# %%
# دليل النواة
# ---------------
#
# في هذا القسم، نوضح بعض العينات المأخوذة من التوزيعات المسبقة واللاحقة للعملية الغاوسية مع نوى مختلفة.
#
# نواة دالة الأساس الشعاعي
# ............................
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

kernel = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# رسم التوزيع المسبق
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("عينات من التوزيع المسبق")

# رسم التوزيع اللاحق
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="الملاحظات")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("عينات من التوزيع اللاحق")

fig.suptitle("نواة دالة الأساس الشعاعي", fontsize=18)
plt.tight_layout()

# %%
print(f"معلمات النواة قبل الملاءمة:\n{kernel})")
print(
    f"معلمات النواة بعد الملاءمة: \n{gpr.kernel_} \n"
    f"احتمالية السجل: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

# %%
# النواة التربيعية النسبية
# .........................
from sklearn.gaussian_process.kernels import RationalQuadratic

kernel = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15))
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("عينات من التوزيع المسبق")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="الملاحظات")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("عينات من التوزيع اللاحق")

fig.suptitle("النواة التربيعية النسبية", fontsize=18)
plt.tight_layout()

# %%
print(f"معلمات النواة قبل الملاءمة:\n{kernel})")
print(
    f"معلمات النواة بعد الملاءمة: \n{gpr.kernel_} \n"
    f"احتمالية السجل: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

# %%
# نواة Exp-Sine-Squared
# .......................
from sklearn.gaussian_process.kernels import ExpSineSquared

kernel = 1.0 * ExpSineSquared(
    length_scale=1.0,
    periodicity=3.0,
    length_scale_bounds=(0.1, 10.0),
    periodicity_bounds=(1.0, 10.0),
)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("عينات من التوزيع المسبق")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="الملاحظات")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("عينات من التوزيع اللاحق")

fig.suptitle("نواة Exp-Sine-Squared", fontsize=18)
plt.tight_layout()

# %%
print(f"معلمات النواة قبل الملاءمة:\n{kernel})")
print(
    f"معلمات النواة بعد الملاءمة: \n{gpr.kernel_} \n"
    f"احتمالية السجل: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

# %%
# نواة Dot-product 
# ..................
from sklearn.gaussian_process.kernels import ConstantKernel, DotProduct

kernel = ConstantKernel(0.1, (0.01, 10.0)) * (
    DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2
)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0, normalize_y=True)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("عينات من التوزيع المسبق")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="الملاحظات")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("عينات من التوزيع اللاحق")

fig.suptitle("نواة Dot-product", fontsize=18)
plt.tight_layout()

# %%
print(f"معلمات النواة قبل الملاءمة:\n{kernel})")
print(
    f"معلمات النواة بعد الملاءمة: \n{gpr.kernel_} \n"
    f"احتمالية السجل: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)

# %%
# نواة Matérn
# ..............
from sklearn.gaussian_process.kernels import Matern

kernel = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel, random_state=0)

fig, axs = plt.subplots(nrows=2, sharex=True, sharey=True, figsize=(10, 8))

# plot prior
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[0])
axs[0].set_title("عينات من التوزيع المسبق")

# plot posterior
gpr.fit(X_train, y_train)
plot_gpr_samples(gpr, n_samples=n_samples, ax=axs[1])
axs[1].scatter(X_train[:, 0], y_train, color="red", zorder=10, label="الملاحظات")
axs[1].legend(bbox_to_anchor=(1.05, 1.5), loc="upper left")
axs[1].set_title("عينات من التوزيع اللاحق")

fig.suptitle("نواة Matérn", fontsize=18)
plt.tight_layout()

# %%
print(f"معلمات النواة قبل الملاءمة:\n{kernel})")
print(
    f"معلمات النواة بعد الملاءمة: \n{gpr.kernel_} \n"
    f"احتمالية السجل: {gpr.log_marginal_likelihood(gpr.kernel_.theta):.3f}"
)


