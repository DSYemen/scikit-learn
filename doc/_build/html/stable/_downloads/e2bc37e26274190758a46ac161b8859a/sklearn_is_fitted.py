"""
========================================
`__sklearn_is_fitted__` كـ API للمطورين
========================================

طريقة `__sklearn_is_fitted__` هي اتفاقية مستخدمة في scikit-learn للتحقق مما إذا كان كائن المُقدر (estimator) قد تم تكييفه (fitted) أم لا. هذه الطريقة يتم تنفيذها عادةً في فئات المُقدرات المخصصة التي يتم بناؤها على فئات القاعدة في scikit-learn مثل `BaseEstimator` أو فئاتها الفرعية.

يجب على المطورين استخدام :func:`~sklearn.utils.validation.check_is_fitted` في بداية جميع الطرق باستثناء `fit`. إذا كانوا بحاجة إلى تخصيص أو تسريع عملية التحقق، يمكنهم تنفيذ طريقة `__sklearn_is_fitted__` كما هو موضح أدناه.

في هذا المثال، يُظهر المُقدر المخصص استخدام طريقة `__sklearn_is_fitted__` ووظيفة فائدة `check_is_fitted` كـ APIs للمطورين. طريقة `__sklearn_is_fitted__` تتحقق من حالة التكييف (fitted) من خلال التحقق من وجود الخاصية `_is_fitted`.

# %%
# مثال على مُقدر مخصص ينفذ مُصنف بسيط
# ------------------------------------------------------------
# هذا الجزء من الكود يُعرّف فئة مُقدر مخصص تسمى `CustomEstimator`
# والتي تمدد كلاً من `BaseEstimator` و `ClassifierMixin` من
# scikit-learn وتُظهر استخدام طريقة `__sklearn_is_fitted__`
# ووظيفة فائدة `check_is_fitted`.
"""
# المؤلفون: مطورو scikit-learn
# معرف الترخيص: BSD-3-Clause

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted


class CustomEstimator(BaseEstimator, ClassifierMixin):
    def __init__(self, parameter=1):
        self.parameter = parameter

    def fit(self, X, y):
        """
        تكييف المُقدر مع بيانات التدريب.
        """
        self.classes_ = sorted(set(y))
        # خاصية مخصصة لتتبع ما إذا كان المُقدر مُكيّفاً
        self._is_fitted = True
        return self

    def predict(self, X):
        """
        إجراء التوقعات

        إذا لم يكن المُقدر مُكيّفاً، فيتم إثارة خطأ NotFittedError
        """
        check_is_fitted(self)
        # منطق التوقع
        predictions = [self.classes_[0]] * len(X)
        return predictions

    def score(self, X, y):
        """
        حساب النتيجة

        إذا لم يكن المُقدر مُكيّفاً، فيتم إثارة خطأ NotFittedError
        """
        check_is_fitted(self)
        # منطق حساب النتيجة
        return 0.5

    def __sklearn_is_fitted__(self):
        """
        التحقق من حالة التكييف وإرجاع قيمة منطقية.
        """
        return hasattr(self, "_is_fitted") and self._is_fitted
