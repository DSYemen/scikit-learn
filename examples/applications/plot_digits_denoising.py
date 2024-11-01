"""
================================
إزالة التشويش من الصور باستخدام PCA النواة
================================

هذا المثال يوضح كيفية استخدام :class:`~sklearn.decomposition.KernelPCA`
لإزالة التشويش من الصور. باختصار، نستفيد من دالة التقريب المُتعلمة أثناء `fit`
لإعادة بناء الصورة الأصلية.

سنقارن النتائج مع إعادة بناء دقيقة باستخدام
:class:`~sklearn.decomposition.PCA`.

سنستخدم مجموعة بيانات أرقام USPS لإعادة إنتاج ما هو مُقدم في القسم 4 من [1]_.

.. rubric:: المراجع

.. [1] `Bakır, Gökhan H., Jason Weston, and Bernhard Schölkopf.
    "Learning to find pre-images."
    Advances in neural information processing systems 16 (2004): 449-456.
    <https://papers.nips.cc/paper/2003/file/ac1ad983e08ad3304a97e147f522747e-Paper.pdf>`_
"""
# المؤلفون: مطورو scikit-learn
# معرف الترخيص: BSD-3-Clause

# %%
# تحميل مجموعة البيانات عبر OpenML
# ---------------------------
#
# مجموعة بيانات أرقام USPS متوفرة في OpenML. نستخدم
# :func:`~sklearn.datasets.fetch_openml` للحصول على هذه المجموعة. بالإضافة إلى ذلك، نقوم
# بتطبيع المجموعة بحيث تكون جميع قيم البكسل في النطاق (0, 1).
import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

X, y = fetch_openml(data_id=41082, as_frame=False, return_X_y=True)
X = MinMaxScaler().fit_transform(X)

# %%
# ستكون الفكرة هي تعلم أساس PCA (مع وبدون نواة) على
# الصور المشوشة، ثم استخدام هذه النماذج لإعادة بناء وإزالة التشويش من هذه
# الصور.
#
# لذلك، نقسم مجموعتنا إلى مجموعة تدريب واختبار مكونة من 1,000
# عينة للتدريب و 100 عينة للاختبار. هذه الصور
# خالية من التشويش وسنستخدمها لتقييم كفاءة طرق إزالة التشويش. بالإضافة إلى ذلك، ننشئ نسخة من
# مجموعة البيانات الأصلية ونضيف تشويشًا غاوسيًا.
#
# فكرة هذا التطبيق هي إظهار أننا يمكننا إزالة التشويش من الصور المشوشة
# من خلال تعلم أساس PCA على بعض الصور غير المشوشة. سنستخدم كل من PCA
# وPCA المعتمد على النواة لحل هذه المشكلة.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=0, train_size=1_000, test_size=100
)

rng = np.random.RandomState(0)
noise = rng.normal(scale=0.25, size=X_test.shape)
X_test_noisy = X_test + noise

noise = rng.normal(scale=0.25, size=X_train.shape)
X_train_noisy = X_train + noise

# %%
# بالإضافة إلى ذلك، سننشئ دالة مساعدة لتقييم إعادة بناء الصورة
# بشكل نوعي من خلال رسم الصور الاختبارية.
import matplotlib.pyplot as plt


def plot_digits(X, title):
    """دالة مساعدة صغيرة لرسم 100 رقم."""
    fig, axs = plt.subplots(nrows=10, ncols=10, figsize=(8, 8))
    for img, ax in zip(X, axs.ravel()):
        ax.imshow(img.reshape((16, 16)), cmap="Greys")
        ax.axis("off")
    fig.suptitle(title, fontsize=24)


# %%
# بالإضافة إلى ذلك، سنستخدم متوسط الخطأ التربيعي (MSE) لتقييم إعادة بناء الصورة
# بشكل كمي.
#
# دعنا نلقي نظرة أولاً لنرى الفرق بين الصور الخالية من التشويش والصور المشوشة.
# سنتحقق من مجموعة الاختبار في هذا الصدد.
plot_digits(X_test, "صور الاختبار غير المشوشة")
plot_digits(
    X_test_noisy, f"صور الاختبار المشوشة\nMSE: {np.mean((X_test - X_test_noisy) ** 2):.2f}"
)

# %%
# تعلم أساس `PCA`
# ---------------------
#
# يمكننا الآن تعلم أساس PCA الخاص بنا باستخدام كل من PCA الخطي وPCA النواة الذي
# يستخدم دالة أساس شعاعية (RBF).
from sklearn.decomposition import PCA, KernelPCA

pca = PCA(n_components=32, random_state=42)
kernel_pca = KernelPCA(
    n_components=400,
    kernel="rbf",
    gamma=1e-3,
    fit_inverse_transform=True,
    alpha=5e-3,
    random_state=42,
)

pca.fit(X_train_noisy)
_ = kernel_pca.fit(X_train_noisy)
pca.fit(X_train_noisy)
_ = kernel_pca.fit(X_train_noisy)

# %%
# إعادة بناء وإزالة تشويش صور الاختبار
# -----------------------------------
#
# الآن، يمكننا تحويل وإعادة بناء مجموعة الاختبار المشوشة. نظرًا لأننا استخدمنا مكونات أقل
# من عدد الخصائص الأصلية، فسنحصل على تقريب
# من المجموعة الأصلية. في الواقع، من خلال إسقاط المكونات التي تفسر التباين
# الأقل في PCA، نأمل في إزالة التشويش. يحدث تفكير مماثل في PCA النواة؛
# ومع ذلك، نتوقع إعادة بناء أفضل لأننا نستخدم نواة غير خطية
# لتعلم أساس PCA ودالة ريدج النواة لتعلم دالة الخريطة.
X_reconstructed_kernel_pca = kernel_pca.inverse_transform(
    kernel_pca.transform(X_test_noisy)
)
X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test_noisy))

# %%
plot_digits(X_test, "صور الاختبار غير المشوشة")
plot_digits(
    X_reconstructed_pca,
    f"إعادة بناء PCA\nMSE: {np.mean((X_test - X_reconstructed_pca) ** 2):.2f}",
)
plot_digits(
    X_reconstructed_kernel_pca,
    (
        "إعادة بناء PCA النواة\n"
        f"MSE: {np.mean((X_test - X_reconstructed_kernel_pca) ** 2):.2f}"
    ),
)

# %%
# لدى PCA متوسط خطأ تربيعي (MSE) أقل من PCA النواة. ومع ذلك، قد لا يفضل التحليل النوعي
# PCA بدلاً من PCA النواة. نلاحظ أن PCA النواة قادر على
# إزالة التشويش الخلفي وتوفير صورة أكثر سلاسة.
#
# ومع ذلك، تجدر الإشارة إلى أن نتائج إزالة التشويش باستخدام PCA النواة
# ستعتمد على المعلمات `n_components` و`gamma` و`alpha`.