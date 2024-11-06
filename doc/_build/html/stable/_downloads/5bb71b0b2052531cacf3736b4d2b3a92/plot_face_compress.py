"""
===========================
مثال على تكميم المتجهات
===========================

يوضح هذا المثال كيف يمكن استخدام :class:`~sklearn.preprocessing.KBinsDiscretizer`
لإجراء تكميم المتجهات على مجموعة من الصور التجريبية، وجه الراكون.
"""

# Authors: The scikit-learn developers
# SPDX-License-Identifier: BSD-3-Clause

# %%
# الصورة الأصلية
# --------------
#
# سنبدأ بتحميل صورة وجه الراكون من SciPy. سنقوم أيضًا بفحص
# بعض المعلومات المتعلقة بالصورة، مثل الشكل ونوع البيانات المستخدم
# لتخزين الصورة.
#
# لاحظ أنه اعتمادًا على إصدار SciPy، نحتاج إلى تعديل الاستيراد
# نظرًا لأن الدالة التي تعيد الصورة ليست موجودة في نفس الوحدة.
# أيضًا، يتطلب SciPy >= 1.10 تثبيت الحزمة `pooch`.
try:  # Scipy >= 1.10
    from sklearn.preprocessing import KBinsDiscretizer
    import matplotlib.pyplot as plt
    from scipy.datasets import face
except ImportError:
    from scipy.misc import face

raccoon_face = face(gray=True)

print(f"البعد الخاص بالصورة هو {raccoon_face.shape}")
print(f"البيانات المستخدمة لترميز الصورة هي من نوع {raccoon_face.dtype}")
print(f"عدد البايتات المستخدمة في الذاكرة هو {raccoon_face.nbytes}")

# %%
# وبالتالي، فإن الصورة هي مصفوفة ثنائية الأبعاد بارتفاع 768 بكسل وعرض 1024 بكسل.
# كل قيمة هي عدد صحيح غير موقع 8 بت، مما يعني أن الصورة مشفرة باستخدام 8 بت لكل بكسل.
# إجمالي استخدام الذاكرة للصورة هو 786 كيلوبايت (1 بايت يساوي 8 بت).
#
# باستخدام عدد صحيح غير موقع 8 بت يعني أن الصورة مشفرة باستخدام 256 لونًا مختلفًا
# من الرمادي، على الأكثر. يمكننا فحص توزيع هذه القيم.

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))

ax[0].imshow(raccoon_face, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("عرض الصورة")
ax[1].hist(raccoon_face.ravel(), bins=256)
ax[1].set_xlabel("قيمة البكسل")
ax[1].set_ylabel("عدد البكسلات")
ax[1].set_title("توزيع قيم البكسل")
_ = fig.suptitle("الصورة الأصلية لوجه الراكون")

# %%
# ضغط عبر تكميم المتجهات
# ----------------------
#
# الفكرة وراء ضغط عبر تكميم المتجهات هي تقليل عدد مستويات الرمادي لتمثيل الصورة.
# على سبيل المثال، يمكننا استخدام 8 قيم بدلاً من 256 قيمة.
# وبالتالي، فهذا يعني أننا يمكن أن نستخدم 3 بت بدلاً من 8 بت لترميز بكسل واحد
# وبالتالي تقليل استخدام الذاكرة بمعامل تقريبًا 2.5. سنناقش لاحقًا حول استخدام الذاكرة.
#
# استراتيجية الترميز
# """""""""""""""""""
#
# يمكن إجراء الضغط باستخدام :class:`~sklearn.preprocessing.KBinsDiscretizer`.
# نحتاج إلى اختيار استراتيجية لتحديد 8 قيم رمادية للتحقيق.
# أبسط استراتيجية هي تحديدها بشكل متساوٍ، مما يتوافق مع ضبط `strategy="uniform"`.
# من الرسم البياني السابق، نعلم أن هذه الاستراتيجية ليست بالضرورة أمثل.


n_bins = 8
encoder = KBinsDiscretizer(
    n_bins=n_bins,
    encode="ordinal",
    strategy="uniform",
    random_state=0,
)
compressed_raccoon_uniform = encoder.fit_transform(raccoon_face.reshape(-1, 1)).reshape(
    raccoon_face.shape
)

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
ax[0].imshow(compressed_raccoon_uniform, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("عرض الصورة")
ax[1].hist(compressed_raccoon_uniform.ravel(), bins=256)
ax[1].set_xlabel("قيمة البكسل")
ax[1].set_ylabel("عدد البكسلات")
ax[1].set_title("توزيع القيم المحققة للبكسل")
_ = fig.suptitle("وجه الراكون المضغوط باستخدام 3 بت واستراتيجية متساوية")

# %%
# نوعيًا، يمكننا ملاحظة بعض المناطق الصغيرة حيث نرى تأثير الضغط
# (مثل الأوراق في الزاوية اليمنى السفلى). لكن بعد كل شيء، الصورة الناتجة
# لا تزال تبدو جيدة.
#
# نلاحظ أن توزيع قيم البكسل تم تعيينه إلى 8 قيم مختلفة. يمكننا التحقق
# من التطابق بين هذه القيم وقيم البكسل الأصلية.

bin_edges = encoder.bin_edges_[0]
bin_center = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
bin_center

# %%
_, ax = plt.subplots()
ax.hist(raccoon_face.ravel(), bins=256)
color = "tab:orange"
for center in bin_center:
    ax.axvline(center, color=color)
    ax.text(center - 10, ax.get_ybound()
            [1] + 100, f"{center:.1f}", color=color)

# %%
# كما ذكرنا سابقًا، الاستراتيجية المتساوية للتحقيق ليست أمثل.
# لاحظ على سبيل المثال أن البكسلات المعينة إلى القيمة 7 ستقوم بترميز
# كمية صغيرة نسبيًا من المعلومات، بينما القيمة المعينة 3 ستمثل كمية
# كبيرة من العدد. يمكننا بدلاً من ذلك استخدام استراتيجية تجميع مثل k-means
# للعثور على تعيين أكثر امتثالًا.

encoder = KBinsDiscretizer(
    n_bins=n_bins,
    encode="ordinal",
    strategy="kmeans",
    random_state=0,
)
compressed_raccoon_kmeans = encoder.fit_transform(raccoon_face.reshape(-1, 1)).reshape(
    raccoon_face.shape
)

fig, ax = plt.subplots(ncols=2, figsize=(12, 4))
ax[0].imshow(compressed_raccoon_kmeans, cmap=plt.cm.gray)
ax[0].axis("off")
ax[0].set_title("عرض الصورة")
ax[1].hist(compressed_raccoon_kmeans.ravel(), bins=256)
ax[1].set_xlabel("قيمة البكسل")
ax[1].set_ylabel("عدد البكسلات")
ax[1].set_title("توزيع قيم البكسل")
_ = fig.suptitle("وجه الراكون المضغوط باستخدام 3 بت واستراتيجية K-means")

# %%
bin_edges = encoder.bin_edges_[0]
bin_center = bin_edges[:-1] + (bin_edges[1:] - bin_edges[:-1]) / 2
bin_center

# %%
_, ax = plt.subplots()
ax.hist(raccoon_face.ravel(), bins=256)
color = "tab:orange"
for center in bin_center:
    ax.axvline(center, color=color)
    ax.text(center - 10, ax.get_ybound()
            [1] + 100, f"{center:.1f}", color=color)

# %%
# العدد في الصناديق الآن أكثر توازنًا ومراكزها لم تعد متساوية المسافات.
# لاحظ أنه يمكننا فرض نفس عدد البكسلات في كل صندوق باستخدام `strategy="quantile"`
# بدلاً من `strategy="kmeans"`.
#
# استخدام الذاكرة
# """"""""""""""""
#
# ذكرنا سابقًا أننا يجب أن نوفر 8 مرات أقل من الذاكرة. دعونا نتحقق من ذلك.

print(
    f"عدد البايتات المستخدمة في الذاكرة هو {compressed_raccoon_kmeans.nbytes}")
print(f"نسبة الضغط: {compressed_raccoon_kmeans.nbytes / raccoon_face.nbytes}")

# %%
# من المدهش جدًا رؤية أن الصورة المضغوطة تستخدم ذاكرة أكثر بـ x8
# من الصورة الأصلية. هذا هو بالضبط عكس ما كنا نتوقعه. السبب يرجع أساسًا
# إلى نوع البيانات المستخدم لترميز الصورة.

print(f"نوع الصورة المضغوطة: {compressed_raccoon_kmeans.dtype}")

# %%
# في الواقع، ناتج :class:`~sklearn.preprocessing.KBinsDiscretizer` هو
# مصفوفة من النوع float64. هذا يعني أنها تستخدم ذاكرة أكثر بـ x8.
# ومع ذلك، نحن نستخدم هذا التمثيل float64 لترميز 8 قيم. في الواقع،
# سنوفر الذاكرة فقط إذا قمنا بتحويل الصورة المضغوطة إلى مصفوفة من الأعداد
# الصحيحة التي تستخدم 3 بت. يمكننا استخدام طريقة `numpy.ndarray.astype`.
# ومع ذلك، لا يوجد تمثيل عدد صحيح بـ 3 بت ولترميز الـ 8 قيم، سنحتاج إلى
# استخدام تمثيل عدد صحيح غير موقع 8 بت أيضًا.
#
# في الممارسة العملية، ملاحظة مكسب في الذاكرة ستتطلب أن تكون الصورة الأصلية
# بتمثيل float64.
