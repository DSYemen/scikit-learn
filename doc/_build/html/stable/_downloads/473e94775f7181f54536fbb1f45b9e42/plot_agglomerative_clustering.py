"""
# التجميع التجميعي مع وبغير بنية
===================================================

هذا المثال يوضح تأثير فرض رسم بياني للاتصال لالتقاط البنية المحلية في البيانات. الرسم البياني هو ببساطة رسم بياني لأقرب 20 جارًا.

هناك ميزتان لفرض الاتصال. أولاً، التجميع مع المصفوفات الاتصالية المتناثرة أسرع بشكل عام.

ثانيًا، عند استخدام مصفوفة الاتصال، فإن الارتباط الفردي والمتوسط والكامل غير مستقرين ويميلون إلى إنشاء عدد قليل من التجمعات التي تنمو بسرعة كبيرة. في الواقع، يحارب الارتباط المتوسط والكامل هذا السلوك التغلغلي من خلال مراعاة جميع المسافات بين التجمعين عند دمجهما (بينما يبالغ الارتباط الفردي في السلوك من خلال مراعاة المسافة الأقصر فقط بين التجمعات). يكسر الرسم البياني للاتصال هذه الآلية للارتباط المتوسط والكامل، مما يجعلها تشبه الارتباط الفردي الهش. هذا التأثير أكثر وضوحًا للرسوم البيانية المتناثرة للغاية (حاول تقليل عدد الجيران في kneighbors_graph) ومع الارتباط الكامل. على وجه الخصوص، فإن وجود عدد صغير جدًا من الجيران في الرسم البياني يفرض هندسة قريبة من هندسة الارتباط الفردي، وهو معروف جيدًا بوجود عدم استقرار التغلغل هذا.
"""
# المؤلفون: مطوري سكايت-ليرن
# معرف SPDX-License: BSD-3-Clause

import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import kneighbors_graph

# توليد بيانات العينة
n_samples = 1500
np.random.seed(0)
t = 1.5 * np.pi * (1 + 3 * np.random.rand(1, n_samples))
x = t * np.cos(t)
y = t * np.sin(t)


X = np.concatenate((x, y))
X += 0.7 * np.random.randn(2, n_samples)
X = X.T

# إنشاء رسم بياني لالتقاط الاتصال المحلي. سيعطي عدد أكبر من الجيران
# مجموعات أكثر تجانسًا بتكلفة وقت الحساب
# الوقت. يعطي عدد كبير جدًا من الجيران أحجام مجموعات موزعة بالتساوي أكثر، ولكن قد لا يفرض البنية الهندسية المحلية
# البيانات
knn_graph = kneighbors_graph(X, 30, include_self=False)

for connectivity in (None, knn_graph):
    for n_clusters in (30, 3):
        plt.figure(figsize=(10, 4))
        for index, linkage in enumerate(("average", "complete", "ward", "single")):
            plt.subplot(1, 4, index + 1)
            model = AgglomerativeClustering(
                linkage=linkage, connectivity=connectivity, n_clusters=n_clusters
            )
            t0 = time.time()
            model.fit(X)
            elapsed_time = time.time() - t0
            plt.scatter(X[:, 0], X[:, 1], c=model.labels_, cmap=plt.cm.nipy_spectral)
            plt.title(
                "linkage=%s\n(time %.2fs)" % (linkage, elapsed_time),
                fontdict=dict(verticalalignment="top"),
            )
            plt.axis("equal")
            plt.axis("off")

            plt.subplots_adjust(bottom=0, top=0.83, wspace=0, left=0, right=1)
            plt.suptitle(
                "n_cluster=%i, connectivity=%r"
                % (n_clusters, connectivity is not None),
                size=17,
            )


plt.show()
