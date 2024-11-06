"""
=====================================
فصل المصدر الأعمى باستخدام FastICA
=====================================

مثال على تقدير المصادر من بيانات مشوشة.

يتم استخدام :ref:`ICA` لتقدير المصادر في ضوء قياسات مشوشة.
تخيل 3 آلات موسيقية تعزف في وقت واحد و 3 ميكروفونات
تسجل الإشارات المختلطة. يتم استخدام ICA لاستعادة المصادر
أي ما يتم عزفه بواسطة كل آلة. الأهم من ذلك، أن PCA يفشل
في استعادة "الآلات" الخاصة بنا لأن الإشارات ذات الصلة تعكس
عمليات غير غاوسية.

"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# إنشاء بيانات نموذجية
# --------------------

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
import numpy as np
from scipy import signal

np.random.seed(0)
n_samples = 2000
time = np.linspace(0, 8, n_samples)

s1 = np.sin(2 * time)  # الإشارة 1: إشارة جيبية
s2 = np.sign(np.sin(3 * time))  # الإشارة 2: إشارة مربعة
s3 = signal.sawtooth(2 * np.pi * time)  # الإشارة 3: إشارة سن المنشار

S = np.c_[s1, s2, s3]
S += 0.2 * np.random.normal(size=S.shape)  # إضافة ضوضاء

S /= S.std(axis=0)  # توحيد البيانات
# خلط البيانات
A = np.array([[1, 1, 1], [0.5, 2, 1.0], [1.5, 1.0, 2.0]])  # مصفوفة الخلط
X = np.dot(S, A.T)  # إنشاء الملاحظات

# %%
# ملاءمة نماذج ICA و PCA
# ----------------------


# حساب ICA
ica = FastICA(n_components=3, whiten="arbitrary-variance")
S_ = ica.fit_transform(X)  # إعادة بناء الإشارات
A_ = ica.mixing_  # الحصول على مصفوفة الخلط المقدرة

# يمكننا "إثبات" أن نموذج ICA ينطبق عن طريق عكس عدم الخلط.
assert np.allclose(X, np.dot(S_, A_.T) + ica.mean_)

# للمقارنة، حساب PCA
pca = PCA(n_components=3)
H = pca.fit_transform(X)  # إعادة بناء الإشارات بناءً على المكونات المتعامدة

# %%
# رسم النتائج
# ------------


plt.figure()

models = [X, S, S_, H]
names = [
    "الملاحظات (إشارة مختلطة)",
    "المصادر الحقيقية",
    "إشارات ICA المستعادة",
    "إشارات PCA المستعادة",
]
colors = ["red", "steelblue", "orange"]

for ii, (model, name) in enumerate(zip(models, names), 1):
    plt.subplot(4, 1, ii)
    plt.title(name)
    for sig, color in zip(model.T, colors):
        plt.plot(sig, color=color)

plt.tight_layout()
plt.show()
