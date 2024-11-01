"""
======================================================================
عرض توضيحي لتجميع هرمي منظم على صورة عملات معدنية
======================================================================

احسب تجزئة صورة ثنائية الأبعاد باستخدام التجميع الهرمي. التجميع مقيد مكانيًا لضمان أن تكون كل منطقة مجزأة قطعة واحدة.
"""
# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

# %%
# توليد البيانات
# -------------

from skimage.data import coins

orig_coins = coins()

# %%
# تغيير حجمها إلى 20% من الحجم الأصلي لتسريع المعالجة
# تطبيق مرشح غاوسي للتنعيم قبل التغيير إلى حجم أصغر
# يقلل من آثار التحجيم.

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.transform import rescale

smoothened_coins = gaussian_filter(orig_coins, sigma=2)
rescaled_coins = rescale(
    smoothened_coins,
    0.2,
    mode="reflect",
    anti_aliasing=False,
)

X = np.reshape(rescaled_coins, (-1, 1))

# %%
# تحديد بنية البيانات
# ----------------------------
#
# البكسلات متصلة بجيرانها.

from sklearn.feature_extraction.image import grid_to_graph

connectivity = grid_to_graph(*rescaled_coins.shape)

# %%
# حساب التجميع
# ------------------

import time as time

from sklearn.cluster import AgglomerativeClustering

print("Compute structured hierarchical clustering...")
st = time.time()
n_clusters = 27  # number of regions
ward = AgglomerativeClustering(
    n_clusters=n_clusters, linkage="ward", connectivity=connectivity
)
ward.fit(X)
label = np.reshape(ward.labels_, rescaled_coins.shape)
print(f"Elapsed time: {time.time() - st:.3f}s")
print(f"Number of pixels: {label.size}")
print(f"Number of clusters: {np.unique(label).size}")

# %%
# عرض النتائج على صورة
# ----------------------------
#
# التجميع التجميعي قادر على تجزئة كل عملة معدنية، ولكن كان علينا
# استخدام "n_cluster" أكبر من عدد العملات المعدنية لأن التجزئة
# تجد منطقة كبيرة في الخلفية.

import matplotlib.pyplot as plt

plt.figure(figsize=(5, 5))
plt.imshow(rescaled_coins, cmap=plt.cm.gray)
for l in range(n_clusters):
    plt.contour(
        label == l,
        colors=[
            plt.cm.nipy_spectral(l / float(n_clusters)),
        ],
    )
plt.axis("off")
plt.show()
