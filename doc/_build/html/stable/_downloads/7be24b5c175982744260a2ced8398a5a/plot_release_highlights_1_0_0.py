# ruff: noqa
"""
=======================================
أبرز ميزات الإصدار 1.0 من scikit-learn
=======================================

.. currentmodule:: sklearn

يسعدنا الإعلان عن إصدار scikit-learn 1.0! لقد كانت المكتبة مستقرة لفترة طويلة، وإصدار الإصدار 1.0 هو اعتراف بذلك وإشارة إلى مستخدمينا. لا يتضمن هذا الإصدار أي تغييرات جذرية باستثناء دورة الاستبعاد المعتادة على مدى إصدارين. بالنسبة للمستقبل، سنبذل قصارى جهدنا للحفاظ على هذا النمط.

يتضمن هذا الإصدار بعض الميزات الرئيسية الجديدة بالإضافة إلى العديد من التحسينات وإصلاح الأخطاء. نقدم أدناه بعض الميزات الرئيسية لهذا الإصدار. **للاطلاع على قائمة شاملة بجميع التغييرات**، يرجى الرجوع إلى :ref:`ملاحظات الإصدار <release_notes_1_0>`.

لتثبيت أحدث إصدار (باستخدام pip)::

    pip install --upgrade scikit-learn

أو باستخدام conda::

    conda install -c conda-forge scikit-learn

"""

##############################################################################
# الحجج الكلمة المفتاحية والحجج الموضعية
# ---------------------------------------------------------
# تعرض واجهة برمجة التطبيقات (API) الخاصة بـ scikit-learn العديد من الوظائف والطرق التي لها العديد من معلمات الإدخال. على سبيل المثال، قبل هذا الإصدار، كان من الممكن إنشاء مثيل لـ :class:`~ensemble.HistGradientBoostingRegressor` كما يلي::
#
#         HistGradientBoostingRegressor("squared_error", 0.1, 100, 31, None,
#             20, 0.0, 255, None, None, False, "auto", "loss", 0.1, 10, 1e-7,
#             0, None)
#
# يتطلب فهم الكود أعلاه من القارئ الذهاب إلى وثائق واجهة برمجة التطبيقات والتحقق من كل معلمة لموضعها ومعناها. لتحسين قابلية قراءة الكود المكتوب بناءً على scikit-learn، يجب على المستخدمين الآن توفير معظم المعلمات بأسمائها، كحجج الكلمة المفتاحية، بدلاً من الحجج الموضعية. على سبيل المثال، سيكون الكود أعلاه::
#
#     HistGradientBoostingRegressor(
#         loss="squared_error",
#         learning_rate=0.1,
#         max_iter=100,
#         max_leaf_nodes=31,
#         max_depth=None,
#         min_samples_leaf=20,
#         l2_regularization=0.0,
#         max_bins=255,
#         categorical_features=None,
#         monotonic_cst=None,
#         warm_start=False,
#         early_stopping="auto",
#         scoring="loss",
#         validation_fraction=0.1,
#         n_iter_no_change=10,
#         tol=1e-7,
#         verbose=0,
#         random_state=None,
#     )
#
# وهو أكثر قابلية للقراءة. تم إيقاف الحجج الموضعية منذ الإصدار 0.23، وسيؤدي الآن إلى إثارة خطأ من النوع ``TypeError``. لا يزال عدد محدود من الحجج الموضعية مسموحًا به في بعض الحالات، على سبيل المثال في :class:`~decomposition.PCA`، حيث ``PCA(10)`` لا يزال مسموحًا به، ولكن ``PCA(10,
# False)`` غير مسموح به.

##############################################################################
# محولات المنحنيات
# ---------------------------------------------------------
# إحدى الطرق لإضافة مصطلحات غير خطية إلى مجموعة ميزات مجموعة بيانات هي توليد
# دوال أساس المنحنيات للسمات المستمرة/العددية باستخدام :class:`~preprocessing.SplineTransformer` الجديد. المنحنيات هي متعددات حدود قطعية،
# يتم معلمتها بدرجة متعددة الحدود ومواضع العقد. ينفذ :class:`~preprocessing.SplineTransformer` أساس المنحنيات من النوع B.
#
# .. figure:: ../linear_model/images/sphx_glr_plot_polynomial_interpolation_001.png
#   :target: ../linear_model/plot_polynomial_interpolation.html
#   :align: center
#
# يوضح الكود التالي المنحنيات في العمل، للحصول على مزيد من المعلومات، يرجى
# الرجوع إلى :ref:`دليل المستخدم <spline_transformer>`.

import numpy as np
from sklearn.preprocessing import SplineTransformer

X = np.arange(5).reshape(5, 1)
spline = SplineTransformer(degree=2, n_knots=3)
spline.fit_transform(X)


##############################################################################
# منظم الكمية
# --------------------------------------------------------------------------
# تقدر عملية الانحدار الكمي الوسيط أو الكميات الأخرى لـ :math:`y`
# المشروطة على :math:`X`، بينما تقدر طريقة المربعات الصغرى العادية (OLS) المتوسط الشرطي.
#
# كنموذج خطي، يعطي :class:`~linear_model.QuantileRegressor` الجديد
# تنبؤات خطية :math:`\hat{y}(w, X) = Xw` للكمية :math:`q`-th،
# :math:`q \in (0, 1)`. يتم بعد ذلك العثور على الأوزان أو المعاملات :math:`w` عن طريق
# مشكلة التقليص التالية:
#
# .. math::
#     \min_{w} {\frac{1}{n_{\text{samples}}}
#     \sum_i PB_q(y_i - X_i w) + \alpha ||w||_1}.
#
# يتكون هذا من خسارة pinball (المعروفة أيضًا باسم الخسارة الخطية)،
# راجع أيضًا :class:`~sklearn.metrics.mean_pinball_loss`،
#
# .. math::
#     PB_q(t) = q \max(t, 0) + (1 - q) \max(-t, 0) =
#     \begin{cases}
#         q t, & t > 0, \\
#         0,    & t = 0, \\
#         (1-q) t, & t < 0
#     \end{cases}
#
# وعقوبة L1 التي يتحكم فيها المعامل "alpha"، مشابهة لـ
# :class:`linear_model.Lasso`.
#
# يرجى التحقق من المثال التالي لمعرفة كيفية عمله، و :ref:`دليل المستخدم <quantile_regression>` للحصول على مزيد من التفاصيل.
#
# .. figure:: ../linear_model/images/sphx_glr_plot_quantile_regression_002.png
#    :target: ../linear_model/plot_quantile_regression.html
#    :align: center
#    :scale: 50%

##############################################################################
# دعم أسماء الميزات
# --------------------------------------------------------------------------
# عندما يتم تمرير مصنف إلى إطار بيانات `pandas'
# <https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe>`_ أثناء
# :term:`fit`، سيقوم المصنف بتعيين سمة `feature_names_in_`
# تحتوي على أسماء الميزات. يرجى ملاحظة أن دعم أسماء الميزات ممكن فقط
# عندما تكون أسماء الأعمدة في إطار البيانات جميعها من النوع سلسلة. يتم استخدام `feature_names_in_`
# للتحقق من أن أسماء أعمدة إطار البيانات الممررة في
# non-:term:`fit`، مثل :term:`predict`، متسقة مع الميزات في
# :term:`fit`:
from sklearn.preprocessing import StandardScaler
import pandas as pd

X = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["a", "b", "c"])
scalar = StandardScaler().fit(X)
scalar.feature_names_in_

# %%
# يتوفر دعم :term:`get_feature_names_out` للمحولات
# التي كان لديها بالفعل `get_feature_names` والمحولات ذات المراسلات من واحد إلى واحد
# بين الإدخال والإخراج مثل
# :class:`~preprocessing.StandardScaler`. سيتم إضافة دعم :term:`get_feature_names_out`
# إلى جميع المحولات الأخرى في الإصدارات المستقبلية. بالإضافة إلى ذلك،
# :meth:`compose.ColumnTransformer.get_feature_names_out` متاح
# لدمج أسماء الميزات لمحولاته:
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

X = pd.DataFrame({"pet": ["dog", "cat", "fish"], "age": [3, 7, 1]})
preprocessor = ColumnTransformer(
    [
        ("numerical", StandardScaler(), ["age"]),
        ("categorical", OneHotEncoder(), ["pet"]),
    ],
    verbose_feature_names_out=False,
).fit(X)

preprocessor.get_feature_names_out()

# %%
# عند استخدام هذا المحول ``preprocessor`` مع خط أنابيب، يتم الحصول على أسماء الميزات
# التي يستخدمها المصنف عن طريق تقطيع واستدعاء
# :term:`get_feature_names_out`:
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

y = [1, 0, 1]
pipe = make_pipeline(preprocessor, LogisticRegression())
pipe.fit(X, y)
pipe[:-1].get_feature_names_out()


##############################################################################
# واجهة برمجة تطبيقات رسم أكثر مرونة
# --------------------------------------------------------------------------
# :class:`metrics.ConfusionMatrixDisplay`،
# :class:`metrics.PrecisionRecallDisplay`، :class:`metrics.DetCurveDisplay`،
# و :class:`inspection.PartialDependenceDisplay` تعرض الآن طريقتين للصف: `from_estimator` و `from_predictions` والتي تسمح للمستخدمين بإنشاء
# رسم بياني معطى التنبؤات أو المصنف. وهذا يعني أن وظائف `plot_*` المقابلة تم إيقافها. يرجى التحقق من المثال الأول
# <sphx_glr_auto_examples_model_selection_plot_confusion_matrix.py>` والمثال الثاني
# <sphx_glr_auto_examples_classification_plot_digits_classification.py>` لمعرفة كيفية استخدام وظائف الرسم الجديدة.

##############################################################################
# SVM من الفئة الواحدة عبر الإنترنت
# --------------------------------------------------------------------------
# تنفذ الفئة الجديدة :class:`~linear_model.SGDOneClassSVM` إصدارًا خطيًا عبر الإنترنت
# من SVM من الفئة الواحدة باستخدام الانحدار التدريجي العشوائي.
# عند دمجها مع تقنيات تقريب النواة،
# يمكن استخدام :class:`~linear_model.SGDOneClassSVM` لتقريب حل
# SVM من الفئة الواحدة المطبق على النواة، المنفذ في :class:`~svm.OneClassSVM`، مع
# تعقيد وقت التجهيز الخطي في عدد العينات. يرجى ملاحظة أن
# تعقيد SVM من الفئة الواحدة المطبق على النواة هو على الأكثر تربيعي في عدد
# العينات. :class:`~linear_model.SGDOneClassSVM` مناسب جيدًا
# لمجموعات البيانات ذات عدد كبير من عينات التدريب (> 10,000) والتي يمكن أن يكون فيها متغير SGD أسرع بعدة رتب من حيث الحجم. يرجى التحقق من هذا
# المثال
# <sphx_glr_auto_examples_miscellaneous_plot_anomaly_comparison.py>` لمعرفة كيفية استخدامه، و :ref:`دليل المستخدم <sgd_online_one_class_svm>` للحصول على مزيد من التفاصيل.
#
# .. figure:: ../miscellaneous/images/sphx_glr_plot_anomaly_comparison_001.png
#    :target: ../miscellaneous/plot_anomaly_comparison.html
#    :align: center

##############################################################################
# نماذج التعزيز التدريجي القائمة على المنحنيات أصبحت مستقرة الآن
# --------------------------------------------------------------------------
# :class:`~sklearn.ensemble.HistGradientBoostingRegressor` و
# :class:`~ensemble.HistGradientBoostingClassifier` لم تعد تجريبية
# ويمكن استيرادها واستخدامها ببساطة كما يلي::
#
#     from sklearn.ensemble import HistGradientBoostingClassifier

##############################################################################
# تحسينات التوثيق الجديدة
# ------------------------------
# يتضمن هذا الإصدار العديد من التحسينات على التوثيق. من بين أكثر من 2100
# طلبات السحب المدمجة، حوالي 800 منها هي تحسينات على توثيقنا.