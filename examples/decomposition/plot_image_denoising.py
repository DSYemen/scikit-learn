"""
=========================================
إزالة تشويش الصور باستخدام تعلم القاموس
=========================================

مثال يقارن تأثير إعادة بناء أجزاء مشوشة
من صورة وجه راكون باستخدام أولاً :ref:`DictionaryLearning` عبر الإنترنت
وطرق تحويل مختلفة.

يتم ملاءمة القاموس على النصف الأيسر المشوه من الصورة، و
يتم استخدامه لاحقًا لإعادة بناء النصف الأيمن. لاحظ أنه يمكن تحقيق أداء أفضل
عن طريق ملاءمة صورة غير مشوهة (أي
بدون تشويش)، ولكننا هنا نبدأ من افتراض أنها غير
متوفرة.

من الممارسات الشائعة لتقييم نتائج إزالة تشويش الصور هي النظر
إلى الفرق بين إعادة البناء والصورة الأصلية. إذا كانت
إعادة البناء مثالية، فسيبدو هذا وكأنه ضوضاء غاوسية.

يمكن ملاحظة من الرسوم البيانية أن نتائج :ref:`omp` مع اثنين
من المعاملات غير الصفرية أقل تحيزًا قليلاً من الاحتفاظ بمعامل واحد فقط
(تبدو الحواف أقل بروزًا). بالإضافة إلى ذلك، فهي أقرب إلى الحقيقة
الأصلية في قاعدة Frobenius.

نتيجة :ref:`least_angle_regression` متحيزة بشكل أقوى:
الفرق يذكرنا بقيمة الكثافة المحلية للصورة الأصلية.

من الواضح أن العتبة ليست مفيدة لإزالة التشويش، ولكنها هنا لإظهار
أنه يمكنها إنتاج مخرجات موحية بسرعة عالية جدًا، وبالتالي تكون مفيدة
للمهام الأخرى مثل تصنيف الكائنات، حيث لا يرتبط الأداء
بالضرورة بالتصور.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# إنشاء صورة مشوهة
# ------------------------
import numpy as np

try:  # Scipy >= 1.10
    from scipy.datasets import face
except ImportError:
    from scipy.misc import face

raccoon_face = face(gray=True)

# تحويل من تمثيل uint8 بقيم بين 0 و 255 إلى
# تمثيل فاصلة عائمة بقيم بين 0 و 1.
raccoon_face = raccoon_face / 255.0

# تقليل العينات لسرعة أعلى
raccoon_face = (
    raccoon_face[::4, ::4]
    + raccoon_face[1::4, ::4]
    + raccoon_face[::4, 1::4]
    + raccoon_face[1::4, 1::4]
)
raccoon_face /= 4.0
height, width = raccoon_face.shape

# تشويه النصف الأيمن من الصورة
print("تشويه الصورة...")
distorted = raccoon_face.copy()
distorted[:, width // 2 :] += 0.075 * np.random.randn(height, width // 2)


# %%
# عرض الصورة المشوهة
# ---------------------------
import matplotlib.pyplot as plt


def show_with_diff(image, reference, title):
    """دالة مساعدة لعرض إزالة التشويش"""
    plt.figure(figsize=(5, 3.3))
    plt.subplot(1, 2, 1)
    plt.title("الصورة")
    plt.imshow(image, vmin=0, vmax=1, cmap=plt.cm.gray, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
    plt.subplot(1, 2, 2)
    difference = image - reference

    plt.title("الفرق (القاعدة: %.2f)" % np.sqrt(np.sum(difference**2)))
    plt.imshow(
        difference, vmin=-0.5, vmax=0.5, cmap=plt.cm.PuOr, interpolation="nearest"
    )
    plt.xticks(())
    plt.yticks(())
    plt.suptitle(title, size=16)
    plt.subplots_adjust(0.02, 0.02, 0.98, 0.79, 0.02, 0.2)


show_with_diff(distorted, raccoon_face, "الصورة المشوهة")


# %%
# استخراج الرقع المرجعية
# ----------------------------
from time import time

from sklearn.feature_extraction.image import extract_patches_2d

# استخراج جميع الرقع المرجعية من النصف الأيسر من الصورة
print("استخراج الرقع المرجعية...")
t0 = time()
patch_size = (7, 7)
data = extract_patches_2d(distorted[:, : width // 2], patch_size)
data = data.reshape(data.shape[0], -1)
data -= np.mean(data, axis=0)
data /= np.std(data, axis=0)
print(f"{data.shape[0]} رقعة مستخرجة في %.2fs." % (time() - t0))


# %%
# تعلم القاموس من الرقع المرجعية
# -------------------------------------------
from sklearn.decomposition import MiniBatchDictionaryLearning

print("تعلم القاموس...")
t0 = time()
dico = MiniBatchDictionaryLearning(
    # زيادة إلى 300 للحصول على نتائج عالية الجودة على حساب أوقات
    # تدريب أبطأ.
    n_components=50,
    batch_size=200,
    alpha=1.0,
    max_iter=10,
)
V = dico.fit(data).components_
dt = time() - t0
print(f"{dico.n_iter_} تكرار / {dico.n_steps_} خطوة في {dt:.2f}.")

plt.figure(figsize=(4.2, 4))
for i, comp in enumerate(V[:100]):
    plt.subplot(10, 10, i + 1)
    plt.imshow(comp.reshape(patch_size), cmap=plt.cm.gray_r, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())
plt.suptitle(
    "قاموس تم تعلمه من رقع الوجه\n"
    + "وقت التدريب %.1fs على %d رقعة" % (dt, len(data)),
    fontsize=16,
)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)


# %%
# استخراج الرقع المشوشة وإعادة بنائها باستخدام القاموس
# ---------------------------------------------------------------
from sklearn.feature_extraction.image import reconstruct_from_patches_2d

print("استخراج الرقع المشوشة... ")
t0 = time()
data = extract_patches_2d(distorted[:, width // 2 :], patch_size)
data = data.reshape(data.shape[0], -1)
intercept = np.mean(data, axis=0)
data -= intercept
print("تم في %.2fs." % (time() - t0))

transform_algorithms = [
    (
        "مطابقة المسار المتعامد\n1 ذرة",
        "omp",
        {"transform_n_nonzero_coefs": 1},
    ),
    (
        "مطابقة المسار المتعامد\n2 ذرة",
        "omp",
        {"transform_n_nonzero_coefs": 2},
    ),
    (
        "انحدار الزاوية الصغرى\n4 ذرات",
        "lars",
        {"transform_n_nonzero_coefs": 4},
    ),
    ("عتبة\n alpha=0.1", "threshold", {"transform_alpha": 0.1}),
]

reconstructions = {}
for title, transform_algorithm, kwargs in transform_algorithms:
    print(title + "...")
    reconstructions[title] = raccoon_face.copy()
    t0 = time()
    dico.set_params(transform_algorithm=transform_algorithm, **kwargs)
    code = dico.transform(data)
    patches = np.dot(code, V)

    patches += intercept
    patches = patches.reshape(len(data), *patch_size)
    if transform_algorithm == "threshold":
        patches -= patches.min()
        patches /= patches.max()
    reconstructions[title][:, width // 2 :] = reconstruct_from_patches_2d(
        patches, (height, width // 2)
    )
    dt = time() - t0
    print("تم في %.2fs." % dt)
    show_with_diff(reconstructions[title], raccoon_face, title + " (الوقت: %.1fs)" % dt)

plt.show()

