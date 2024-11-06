"""
==========================
مربعات أقل غير سالبة
==========================

في هذا المثال، نقوم بملاءمة نموذج خطي مع قيود إيجابية على
معاملات الانحدار ومقارنة المعاملات المقدرة مع الانحدار الخطي الكلاسيكي.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import r2_score

# %%
# توليد بعض البيانات العشوائية
np.random.seed(42)

n_samples, n_features = 200, 50
X = np.random.randn(n_samples, n_features)
true_coef = 3 * np.random.randn(n_features)
# عتبة المعاملات لجعلها غير سالبة
true_coef[true_coef < 0] = 0
y = np.dot(X, true_coef)

# إضافة بعض الضوضاء
y += 5 * np.random.normal(size=(n_samples,))

# %%
# تقسيم البيانات إلى مجموعة تدريب ومجموعة اختبار
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)

# %%
# ملاءمة المربعات الأقل غير السالبة.
from sklearn.linear_model import LinearRegression

reg_nnls = LinearRegression(positive=True)
y_pred_nnls = reg_nnls.fit(X_train, y_train).predict(X_test)
r2_score_nnls = r2_score(y_test, y_pred_nnls)
print("NNLS R2 score", r2_score_nnls)

# %%
# ملاءمة OLS.
reg_ols = LinearRegression()
y_pred_ols = reg_ols.fit(X_train, y_train).predict(X_test)
r2_score_ols = r2_score(y_test, y_pred_ols)
print("OLS R2 score", r2_score_ols)


# %%
# عند مقارنة معاملات الانحدار بين OLS و NNLS، يمكننا ملاحظة
# أنها مرتبطة ارتباطًا وثيقًا (الخط المتقطع هو علاقة الهوية)،
# ولكن القيود غير السالبة تقلص بعضها إلى 0.
# المربعات الأقل غير السالبة تعطي نتائج متفرقة بشكل متأصل.

fig, ax = plt.subplots()
ax.plot(reg_ols.coef_, reg_nnls.coef_, linewidth=0, marker=".")

low_x, high_x = ax.get_xlim()
low_y, high_y = ax.get_ylim()
low = max(low_x, low_y)
high = min(high_x, high_y)
ax.plot([low, high], [low, high], ls="--", c=".3", alpha=0.5)
ax.set_xlabel("معاملات الانحدار OLS", fontweight="bold")
ax.set_ylabel("معاملات الانحدار NNLS", fontweight="bold")