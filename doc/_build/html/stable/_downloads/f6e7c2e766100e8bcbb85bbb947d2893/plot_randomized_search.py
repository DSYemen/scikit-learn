"""
=========================================================================
مقارنة البحث العشوائي والبحث الشبكي لتقدير فرط المعلمات
=========================================================================

قارن بين البحث العشوائي والبحث الشبكي لتحسين فرط معلمات SVM الخطي مع التدريب SGD.
يتم البحث عن جميع المعلمات التي تؤثر على التعلم في نفس الوقت
(باستثناء عدد المعلمات، والذي يمثل مفاضلة بين الوقت والجودة).

يستكشف البحث العشوائي والبحث الشبكي نفس مساحة المعلمات بالضبط. النتيجة في إعدادات المعلمات متشابهة جدًا، في حين أن وقت التشغيل للبحث العشوائي أقل بشكل كبير.

قد يكون الأداء أسوأ قليلاً بالنسبة للبحث العشوائي، ومن المحتمل
أن يكون ذلك بسبب تأثير الضوضاء ولن ينتقل إلى مجموعة اختبار محفوظة.

لاحظ أنه في الممارسة العملية، لن يتم البحث عن هذا العدد الكبير من المعلمات المختلفة
في نفس الوقت باستخدام البحث الشبكي، ولكن سيتم اختيار المعلمات التي تعتبر الأكثر أهمية فقط.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

from time import time

import numpy as np
import scipy.stats as stats

from sklearn.datasets import load_digits
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# الحصول على بعض البيانات
X, y = load_digits(return_X_y=True, n_class=3)

# بناء مصنف
clf = SGDClassifier(loss="hinge", penalty="elasticnet", fit_intercept=True)


# دالة مساعدة للإبلاغ عن أفضل النتائج
def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("نموذج مع الترتيب: {0}".format(i))
            print(
                "متوسط درجة التحقق: {0:.3f} (الانحراف المعياري: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("المعلمات: {0}".format(results["params"][candidate]))
            print("")


# تحديد المعلمات والتوزيعات للعينة منها
param_dist = {
    "average": [True, False],
    "l1_ratio": stats.uniform(0, 1),
    "alpha": stats.loguniform(1e-2, 1e0),
}

# تشغيل البحث العشوائي
n_iter_search = 15
random_search = RandomizedSearchCV(
    clf, param_distributions=param_dist, n_iter=n_iter_search
)

start = time()
random_search.fit(X, y)
print(
    "استغرق البحث العشوائي RandomizedSearchCV %.2f ثانية لـ %d إعدادات المعلمات المرشحة."
    % ((time() - start), n_iter_search)
)
report(random_search.cv_results_)

# استخدام شبكة كاملة عبر جميع المعلمات
param_grid = {
    "average": [True, False],
    "l1_ratio": np.linspace(0, 1, num=10),
    "alpha": np.power(10, np.arange(-2, 1, dtype=float)),
}

# تشغيل البحث الشبكي
grid_search = GridSearchCV(clf, param_grid=param_grid)
start = time()
grid_search.fit(X, y)

print(
    "استغرق البحث الشبكي GridSearchCV %.2f ثانية لـ %d إعدادات المعلمات المرشحة."
    % (time() - start, len(grid_search.cv_results_["params"]))
)
report(grid_search.cv_results_)