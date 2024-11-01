"""
تعلم القاموس عبر الإنترنت لأجزاء الوجوه
=================================================

يستخدم هذا المثال مجموعة كبيرة من الوجوه لتعلم مجموعة من الصور مقاس 20x20 التي تشكل الوجوه.

من وجهة نظر البرمجة، إنه مثير للاهتمام لأنه يظهر كيفية استخدام واجهة برمجة التطبيقات عبر الإنترنت لـ scikit-learn لمعالجة مجموعة كبيرة جدًا من البيانات على شكل أجزاء. الطريقة التي نتبعها هي أننا نحمل صورة في كل مرة ونستخرج عشوائيًا 50 رقعة من هذه الصورة. بمجرد أن نكون قد جمعنا 500 من هذه الرقع (باستخدام 10 صور)، فإننا نستخدم طريقة
:func:`~sklearn.cluster.MiniBatchKMeans.partial_fit`
للكائن KMeans عبر الإنترنت، MiniBatchKMeans.

يسمح الإعداد المفصل على MiniBatchKMeans برؤية أنه يتم إعادة تعيين بعض المجموعات أثناء المكالمات المتتالية لـ
partial-fit. هذا لأن عدد الرقع التي تمثلها أصبح منخفضًا للغاية، ومن الأفضل اختيار مجموعة جديدة عشوائية.
"""
# المؤلفون: مطوري scikit-learn
# معرف SPDX-License: BSD-3-Clause

# %%
# تحميل البيانات
# -------------

from sklearn import datasets

faces = datasets.fetch_olivetti_faces()

# %%
# تعلم قاموس الصور
# --------------------

import time

import numpy as np

from sklearn.cluster import MiniBatchKMeans
from sklearn.feature_extraction.image import extract_patches_2d

print("Learning the dictionary... ")
rng = np.random.RandomState(0)
kmeans = MiniBatchKMeans(n_clusters=81, random_state=rng, verbose=True, n_init=3)
patch_size = (20, 20)

buffer = []
t0 = time.time()

# جزء التعلم عبر الإنترنت: دورة عبر مجموعة البيانات الكاملة 6 مرات
index = 0
for _ in range(6):
    for img in faces.images:
        data = extract_patches_2d(img, patch_size, max_patches=50, random_state=rng)
        data = np.reshape(data, (len(data), -1))
        buffer.append(data)
        index += 1
        if index % 10 == 0:
            data = np.concatenate(buffer, axis=0)
            data -= np.mean(data, axis=0)
            data /= np.std(data, axis=0)
            kmeans.partial_fit(data)
            buffer = []
        if index % 100 == 0:
            print("Partial fit of %4i out of %i" % (index, 6 * len(faces.images)))

dt = time.time() - t0
print("done in %.2fs." % dt)

# %%
# رسم النتائج
# --------------

import matplotlib.pyplot as plt

plt.figure(figsize=(4.2, 4))
for i, patch in enumerate(kmeans.cluster_centers_):
    plt.subplot(9, 9, i + 1)
    plt.imshow(patch.reshape(patch_size), cmap=plt.cm.gray, interpolation="nearest")
    plt.xticks(())
    plt.yticks(())


plt.suptitle(
    "Patches of faces\nTrain time %.1fs on %d patches" % (dt, 8 * len(faces.images)),
    fontsize=16,
)
plt.subplots_adjust(0.08, 0.02, 0.92, 0.85, 0.08, 0.23)

plt.show()
