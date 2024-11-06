"""
===================================
أمثلة على استخدام `FrozenEstimator`
===================================

يعرض هذا المثال بعض حالات الاستخدام لـ :class:`~sklearn.frozen.FrozenEstimator`.

:class:`~sklearn.frozen.FrozenEstimator` هي فئة مساعدة تسمح بتجميد
مصنف مُدرَّب. هذا مفيد، على سبيل المثال، عندما نريد تمرير مصنف مُدرَّب
إلى ميتا-مصنف، مثل :class:`~sklearn.model_selection.FixedThresholdClassifier`
دون السماح للميتا-مصنف بإعادة تدريب المصنف.
"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

# %%
# تعيين عتبة قرار لمصنف مُدرَّب مسبقًا
# --------------------------------------------------------
# تستخدم المصنفات المُدرَّبة في سكايلرن عتبة قرار تعسفية لاتخاذ القرار
# حول الفئة التي تنتمي إليها العينة المعطاة. عتبة القرار هي إما `0.0` على القيمة التي
# تعيدها :term:`decision_function`، أو `0.5` على الاحتمال الذي تعيده
# :term:`predict_proba`.
#
# ومع ذلك، قد يرغب المرء في تعيين عتبة قرار مخصصة. يمكننا القيام بذلك عن طريق
# استخدام :class:`~sklearn.model_selection.FixedThresholdClassifier` وتغليف
# المصنف بـ :class:`~sklearn.frozen.FrozenEstimator`.
from sklearn.datasets import make_classification
from sklearn.frozen import FrozenEstimator
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import FixedThresholdClassifier, train_test_split

X, y = make_classification(n_samples=1000, random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
classifier = LogisticRegression().fit(X_train, y_train)

print(
    "Probability estimates for three data points:\n"
    f"{classifier.predict_proba(X_test[-3:]).round(3)}"
)
print(
    "Predicted class for the same three data points:\n"
    f"{classifier.predict(X_test[-3:])}"
)

# %%
# الآن تخيل أنك تريد تعيين عتبة قرار مختلفة على تقديرات الاحتمال. يمكننا القيام بذلك عن طريق
# تغليف المصنف بـ :class:`~sklearn.frozen.FrozenEstimator` وتمريره إلى
# :class:`~sklearn.model_selection.FixedThresholdClassifier`.

threshold_classifier = FixedThresholdClassifier(
    estimator=FrozenEstimator(classifier), threshold=0.9
)

# %%
# لاحظ أنه في قطعة الكود أعلاه، لا يؤدي استدعاء `fit` على
# :class:`~sklearn.model_selection.FixedThresholdClassifier` إلى إعادة تدريب
# المصنف الأساسي.
#
# الآن، دعنا نرى كيف تغيرت التوقعات فيما يتعلق بعتبة الاحتمال.
print(
    "Probability estimates for three data points with FixedThresholdClassifier:\n"
    f"{threshold_classifier.predict_proba(X_test[-3:]).round(3)}"
)
print(
    "Predicted class for the same three data points with FixedThresholdClassifier:\n"
    f"{threshold_classifier.predict(X_test[-3:])}"
)

# %%
# نرى أن تقديرات الاحتمال تبقى كما هي، ولكن نظرًا لاستخدام عتبة قرار مختلفة،
# فإن الفئات المتوقعة مختلفة.
#
# يرجى الرجوع إلى
# :ref:`sphx_glr_auto_examples_model_selection_plot_cost_sensitive_learning.py`
# لمعرفة المزيد عن التعلم الحساس للتكلفة وضبط عتبة القرار.

# %%
# معايرة مصنف مُدرَّب مسبقًا
# --------------------------------------
# يمكنك استخدام :class:`~sklearn.frozen.FrozenEstimator` لمعايرة مصنف مُدرَّب مسبقًا
# باستخدام :class:`~sklearn.calibration.CalibratedClassifierCV`.
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

calibrated_classifier = CalibratedClassifierCV(
    estimator=FrozenEstimator(classifier)
).fit(X_train, y_train)

prob_pos_clf = classifier.predict_proba(X_test)[:, 1]
clf_score = brier_score_loss(y_test, prob_pos_clf)
print(f"No calibration: {clf_score:.3f}")

prob_pos_calibrated = calibrated_classifier.predict_proba(X_test)[:, 1]
calibrated_score = brier_score_loss(y_test, prob_pos_calibrated)
print(f"With calibration: {calibrated_score:.3f}")