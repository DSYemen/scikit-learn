"""
=================================
مقارنة بين BIRCH و MiniBatchKMeans
=================================

يقارن هذا المثال توقيت BIRCH (مع وبدون خطوة التجميع العالمي) و MiniBatchKMeans على مجموعة بيانات اصطناعية تحتوي على 25,000 عينة و2 من الميزات التي تم إنشاؤها باستخدام make_blobs.

كل من ``MiniBatchKMeans`` و ``BIRCH`` هي خوارزميات قابلة للتطوير للغاية ويمكنها العمل بكفاءة على مئات الآلاف أو حتى الملايين من نقاط البيانات. لقد اخترنا تحديد حجم مجموعة البيانات لهذا المثال للحفاظ على استخدام موارد التكامل المستمر لدينا معقولًا، ولكن القارئ المهتم قد يستمتع بتحرير هذا النص البرمجي لإعادة تشغيله بقيمة أكبر لـ `n_samples`.

إذا تم تعيين ``n_clusters`` إلى None، يتم تقليل البيانات من 25,000 عينة إلى مجموعة من 158 مجموعة. يمكن اعتبار هذا كخطوة ما قبل المعالجة قبل خطوة التجميع (العالمي) النهائي التي تقلل هذه المجموعات 158 إلى 100 مجموعة.
"""
# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

from itertools import cycle
from time import time

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from joblib import cpu_count

from sklearn.cluster import Birch, MiniBatchKMeans
from sklearn.datasets import make_blobs

# توليد مراكز للكتل بحيث تشكل شبكة 10X10.
xx = np.linspace(-22, 22, 10)
yy = np.linspace(-22, 22, 10)
xx, yy = np.meshgrid(xx, yy)
n_centers = np.hstack((np.ravel(xx)[:, np.newaxis], np.ravel(yy)[:, np.newaxis]))

# توليد كتل للمقارنة بين MiniBatchKMeans و BIRCH.
X, y = make_blobs(n_samples=25000, centers=n_centers, random_state=0)

# استخدم جميع الألوان التي يوفرها matplotlib بشكل افتراضي.
colors_ = cycle(colors.cnames.keys())

fig = plt.figure(figsize=(12, 4))
fig.subplots_adjust(left=0.04, right=0.98, bottom=0.1, top=0.9)

# حساب التجميع باستخدام BIRCH مع وبدون خطوة التجميع النهائي
# والتخطيط.
birch_models = [
    Birch(threshold=1.7, n_clusters=None),
    Birch(threshold=1.7, n_clusters=100),
]
final_step = ["بدون التجميع العالمي", "مع التجميع العالمي"]

for ind, (birch_model, info) in enumerate(zip(birch_models, final_step)):
    t = time()
    birch_model.fit(X)
    print("BIRCH %s كخطوة نهائية استغرقت %0.2f ثانية" % (info, (time() - t)))

    # عرض النتيجة
    labels = birch_model.labels_
    centroids = birch_model.subcluster_centers_
    n_clusters = np.unique(labels).size
    print("n_clusters : %d" % n_clusters)

    ax = fig.add_subplot(1, 3, ind + 1)
    for this_centroid, k, col in zip(centroids, range(n_clusters), colors_):
        mask = labels == k
        ax.scatter(X[mask, 0], X[mask, 1], c="w", edgecolor=col, marker=".", alpha=0.5)
        if birch_model.n_clusters is None:
            ax.scatter(this_centroid[0], this_centroid[1], marker="+", c="k", s=25)
    ax.set_ylim([-25, 25])
    ax.set_xlim([-25, 25])
    ax.set_autoscaley_on(False)
    ax.set_title("BIRCH %s" % info)

# حساب التجميع باستخدام MiniBatchKMeans.
mbk = MiniBatchKMeans(
    init="k-means++",
    n_clusters=100,
    batch_size=256 * cpu_count(),
    n_init=10,
    max_no_improvement=10,
    verbose=0,
    random_state=0,
)
t0 = time()
mbk.fit(X)
t_mini_batch = time() - t0
print("الوقت المستغرق لتشغيل MiniBatchKMeans %0.2f ثانية" % t_mini_batch)
mbk_means_labels_unique = np.unique(mbk.labels_)

ax = fig.add_subplot(1, 3, 3)
for this_centroid, k, col in zip(mbk.cluster_centers_, range(n_clusters), colors_):
    mask = mbk.labels_ == k
    ax.scatter(X[mask, 0], X[mask, 1], marker=".", c="w", edgecolor=col, alpha=0.5)
    ax.scatter(this_centroid[0], this_centroid[1], marker="+", c="k", s=25)
ax.set_xlim([-25, 25])
ax.set_ylim([-25, 25])
ax.set_title("MiniBatchKMeans")
ax.set_autoscaley_on(False)
plt.show()