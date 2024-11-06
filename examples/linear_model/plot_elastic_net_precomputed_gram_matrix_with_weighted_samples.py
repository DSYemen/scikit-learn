"""
==========================================================================
تناسب شبكة مرنة مع مصفوفة جرام مسبقة الحساب وعينات مرجحة
==========================================================================

يوضح المثال التالي كيفية حساب مصفوفة جرام مسبقًا
مع استخدام عينات مرجحة مع :class:`~sklearn.linear_model.ElasticNet`.

إذا تم استخدام عينات مرجحة، يجب أن يتم ترتيب مصفوفة التصميم ثم إعادة قياسها بواسطة الجذر التربيعي لمتجه الأوزان قبل حساب مصفوفة جرام.

.. note::
  يتم إعادة قياس متجه `sample_weight` أيضًا ليجمع إلى `n_samples`، راجع التوثيق لمتغير `sample_weight` في
  :meth:`~sklearn.linear_model.ElasticNet.fit`.

"""

# المؤلفون: مطوري سكايلرن
# معرف الترخيص: BSD-3-Clause

# %%
# دعنا نبدأ بتحميل مجموعة البيانات وإنشاء بعض أوزان العينات.
import numpy as np

from sklearn.datasets import make_regression

rng = np.random.RandomState(0)

n_samples = int(1e5)
X, y = make_regression(n_samples=n_samples, noise=0.5, random_state=rng)

sample_weight = rng.lognormal(size=n_samples)
# قم بتطبيع أوزان العينات
normalized_weights = sample_weight * (n_samples / (sample_weight.sum()))

# %%
# لتناسب الشبكة المرنة باستخدام خيار `precompute` مع أوزان العينات، يجب علينا أولاً ترتيب مصفوفة التصميم، وإعادة قياسها بواسطة الأوزان
# الموحدة قبل حساب مصفوفة جرام.
X_offset = np.average(X, axis=0, weights=normalized_weights)
X_centered = X - np.average(X, axis=0, weights=normalized_weights)
X_scaled = X_centered * np.sqrt(normalized_weights)[:, np.newaxis]
gram = np.dot(X_scaled.T, X_scaled)

# %%
# يمكننا الآن المتابعة بالتناسب. يجب أن نمرر مصفوفة التصميم المركزية إلى
# `fit` وإلا سيقوم مقدر الشبكة المرنة بالكشف عن أنها غير مركزة
# وسيقوم بتجاهل مصفوفة جرام التي قمنا بتمريرها. ومع ذلك، إذا قمنا بتمرير مصفوفة التصميم المقياس، فإن كود المعالجة المسبقة سيقوم بإعادة قياسها بشكل خاطئ للمرة الثانية.
from sklearn.linear_model import ElasticNet

lm = ElasticNet(alpha=0.01, precompute=gram)
lm.fit(X_centered, y, sample_weight=normalized_weights)