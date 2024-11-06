"""
==============================
Lasso على البيانات الكثيفة والمتفرقة
==============================

نبين أن linear_model.Lasso يوفر نفس النتائج للبيانات الكثيفة والمتفرقة
وأن السرعة تتحسن في حالة البيانات المتفرقة.

"""

# المؤلفون: مطوري scikit-learn
# معرف الترخيص: BSD-3-Clause

from time import time

from scipy import linalg, sparse

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso

# %%
# مقارنة تنفيذي Lasso على البيانات الكثيفة
# -----------------------------------------------------
#
# ننشئ مشكلة انحدار خطي مناسبة لـ Lasso،
# أي بمعنى، مع وجود ميزات أكثر من العينات. ثم نقوم بتخزين مصفوفة البيانات
# في كل من التنسيق الكثيف (العادي) والمتفرق، ونقوم بتدريب Lasso على
# كل منهما. نحسب وقت التشغيل لكل منهما ونتحقق من أنهما تعلما
# نفس النموذج عن طريق حساب المعيار الإقليدي لفرق
# المعاملات التي تعلموها. نظرًا لأن البيانات كثيفة، نتوقع وقت تشغيل أفضل
# مع تنسيق البيانات الكثيفة.

X, y = make_regression(n_samples=200, n_features=5000, random_state=0)
# إنشاء نسخة من X بتنسيق متفرق
X_sp = sparse.coo_matrix(X)

alpha = 1
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=1000)

t0 = time()
sparse_lasso.fit(X_sp, y)
print(f"Sparse Lasso done in {(time() - t0):.3f}s")

t0 = time()
dense_lasso.fit(X, y)
print(f"Dense Lasso done in {(time() - t0):.3f}s")

# مقارنة معاملات الانحدار
coeff_diff = linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_)
print(f"Distance between coefficients : {coeff_diff:.2e}")

#
# %%
# مقارنة تنفيذي Lasso على البيانات المتفرقة
# ------------------------------------------------------
#
# نجعل المشكلة السابقة متفرقة عن طريق استبدال جميع القيم الصغيرة بـ 0
# ونقوم بنفس المقارنات كما هو موضح أعلاه. نظرًا لأن البيانات أصبحت متفرقة الآن، فإننا
# نتوقع أن يكون التنفيذ الذي يستخدم تنسيق البيانات المتفرقة أسرع.

# إنشاء نسخة من البيانات السابقة
Xs = X.copy()
# جعل Xs متفرقة عن طريق استبدال القيم الأقل من 2.5 بـ 0s
Xs[Xs < 2.5] = 0.0
# إنشاء نسخة من Xs بتنسيق متفرق
Xs_sp = sparse.coo_matrix(Xs)
Xs_sp = Xs_sp.tocsc()

# حساب نسبة المعاملات غير الصفرية في مصفوفة البيانات
print(f"Matrix density : {(Xs_sp.nnz / float(X.size) * 100):.3f}%")

alpha = 0.1
sparse_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)
dense_lasso = Lasso(alpha=alpha, fit_intercept=False, max_iter=10000)

t0 = time()
sparse_lasso.fit(Xs_sp, y)
print(f"Sparse Lasso done in {(time() - t0):.3f}s")

t0 = time()
dense_lasso.fit(Xs, y)
print(f"Dense Lasso done in  {(time() - t0):.3f}s")

# مقارنة معاملات الانحدار
coeff_diff = linalg.norm(sparse_lasso.coef_ - dense_lasso.coef_)
print(f"Distance between coefficients : {coeff_diff:.2e}")

# %%