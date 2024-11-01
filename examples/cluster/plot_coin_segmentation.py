"""
تم تقسيم صورة العملات اليونانية إلى مناطق
================================================

يستخدم هذا المثال :ref:`spectral_clustering` على رسم بياني تم إنشاؤه من
الفرق بين البكسلات في صورة لتقسيم هذه الصورة إلى مناطق متعددة
جزئيا متجانسة.

هذه العملية (التجميع الطيفي على صورة) هي حل تقريبي فعال
لإيجاد القطع الرسومية المعيارية.

هناك ثلاثة خيارات لتعيين التصنيفات:

* 'kmeans' التجميع الطيفي يجمع العينات في مساحة التضمين
  باستخدام خوارزمية kmeans
* 'discrete' تبحث بشكل تكراري عن أقرب تقسيم
  مساحة إلى مساحة تضمين التجميع الطيفي.
* 'cluster_qr' يقوم بتعيين التصنيفات باستخدام تحليل QR مع التبديل
  الذي يحدد التقسيم مباشرة في مساحة التضمين.
"""
# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

import time

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.data import coins
from skimage.transform import rescale

from sklearn.cluster import spectral_clustering
from sklearn.feature_extraction import image

# تحميل العملات كصفيف numpy
orig_coins = coins()

# تغيير حجمه إلى 20% من الحجم الأصلي لتسريع المعالجة
# تطبيق مرشح غاوسي للتنعيم قبل تغيير الحجم
# يقلل من آثار التحجيم.
smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(smoothened_coins, 0.2, mode="reflect", anti_aliasing=False)

# تحويل الصورة إلى رسم بياني مع قيمة التدرج على
# الحواف.
graph = image.img_to_graph(rescaled_coins)

# خذ دالة متناقصة من التدرج: أسية
# كلما كان beta أصغر، كلما كانت القطعة مستقلة أكثر عن
# الصورة الفعلية. بالنسبة لـ beta=1، تكون القطعة قريبة من voronoi
beta = 10
eps = 1e-6
graph.data = np.exp(-beta * graph.data / graph.data.std()) + eps

# يجب اختيار عدد المناطق المقسمة يدويًا.
# الإصدار الحالي من 'spectral_clustering' لا يدعم تحديد
# عدد المجموعات ذات الجودة الجيدة تلقائيًا.
n_regions = 26

# %%
# حساب وعرض المناطق الناتجة

# حساب بعض المتجهات الذاتية الإضافية قد يسرع eigen_solver.
# قد تستفيد جودة التجميع الطيفي أيضًا من طلب
# مناطق إضافية للتقسيم.
n_regions_plus = 3

# تطبيق التجميع الطيفي باستخدام eigen_solver='arpack' الافتراضي.
# يمكن استخدام أي محقق تم تنفيذه: eigen_solver='arpack'، 'lobpcg'، أو 'amg'.
# اختيار eigen_solver='amg' يتطلب حزمة إضافية تسمى 'pyamg'.
# يتم تحديد جودة التقسيم وسرعة الحسابات بشكل أساسي
# من خلال اختيار المحقق وقيمة التسامح 'eigen_tol'.
# TODO: يبدو أن تغيير eigen_tol ليس له تأثير على 'lobpcg' و 'amg' #21243.
for assign_labels in ("kmeans", "discretize", "cluster_qr"):
    t0 = time.time()
    labels = spectral_clustering(
        graph,
        n_clusters=(n_regions + n_regions_plus),
        eigen_tol=1e-7,
        assign_labels=assign_labels,
        random_state=42,
    )

    t1 = time.time()
    labels = labels.reshape(rescaled_coins.shape)
    plt.figure(figsize=(5, 5))
    plt.imshow(rescaled_coins, cmap=plt.cm.gray)

    plt.xticks(())
    plt.yticks(())
    title = "Spectral clustering: %s, %.2fs" % (assign_labels, (t1 - t0))
    print(title)
    plt.title(title)
    for l in range(n_regions):
        colors = [plt.cm.nipy_spectral((l + 4) / float(n_regions + 4))]
        plt.contour(labels == l, colors=colors)
        # لعرض المقاطع الفردية كما تظهر قم بالتعليق في plt.pause(0.5)
plt.show()

# TODO: بعد دمج #21194 وإصلاح #21243، تحقق من أفضل محقق
# هو eigen_solver='arpack'، 'lobpcg'، أو 'amg' و eigen_tol
# بشكل صريح في هذا المثال.