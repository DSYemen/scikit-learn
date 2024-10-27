
.. _array_api_supported:

دعم المدخلات المتوافقة مع `Array API`
=========================================

المقدرات والأدوات الأخرى في scikit-learn التي تدعم المدخلات المتوافقة مع واجهة برمجة تطبيقات المصفوفة.

المقدرات
----------

- :class:`decomposition.PCA` (مع `svd_solver="full"`،
  `svd_solver="randomized"` و `power_iteration_normalizer="QR"`)
- :class:`linear_model.Ridge` (مع `solver="svd"`)
- :class:`discriminant_analysis.LinearDiscriminantAnalysis` (مع `solver="svd"`)
- :class:`preprocessing.KernelCenterer`
- :class:`preprocessing.MaxAbsScaler`
- :class:`preprocessing.MinMaxScaler`
- :class:`preprocessing.Normalizer`

المقدرات الوصفية
-------------------

المقدرات الوصفية التي تقبل مدخلات واجهة برمجة تطبيقات المصفوفة بشرط أن يقوم المقدر الأساسي بذلك أيضًا:

- :class:`model_selection.GridSearchCV`
- :class:`model_selection.RandomizedSearchCV`
- :class:`model_selection.HalvingGridSearchCV`
- :class:`model_selection.HalvingRandomSearchCV`

المقاييس
---------

- :func:`sklearn.metrics.cluster.entropy`
- :func:`sklearn.metrics.accuracy_score`
- :func:`sklearn.metrics.d2_tweedie_score`
- :func:`sklearn.metrics.max_error`
- :func:`sklearn.metrics.mean_absolute_error`
- :func:`sklearn.metrics.mean_absolute_percentage_error`
- :func:`sklearn.metrics.mean_gamma_deviance`
- :func:`sklearn.metrics.mean_poisson_deviance` (يتطلب `تمكين دعم واجهة برمجة تطبيقات المصفوفة لـ SciPy <https://docs.scipy.org/doc/scipy/dev/api-dev/array_api.html#using-array-api-standard-support>`_)
- :func:`sklearn.metrics.mean_squared_error`
- :func:`sklearn.metrics.mean_tweedie_deviance`
- :func:`sklearn.metrics.pairwise.additive_chi2_kernel`
- :func:`sklearn.metrics.pairwise.chi2_kernel`
- :func:`sklearn.metrics.pairwise.cosine_similarity`
- :func:`sklearn.metrics.pairwise.cosine_distances`
- :func:`sklearn.metrics.pairwise.euclidean_distances` (انظر :ref:`device_support_for_float64`)
- :func:`sklearn.metrics.pairwise.linear_kernel`
- :func:`sklearn.metrics.pairwise.paired_cosine_distances`
- :func:`sklearn.metrics.pairwise.paired_euclidean_distances`
- :func:`sklearn.metrics.pairwise.polynomial_kernel`
- :func:`sklearn.metrics.pairwise.rbf_kernel` (انظر :ref:`device_support_for_float64`)
- :func:`sklearn.metrics.pairwise.sigmoid_kernel`
- :func:`sklearn.metrics.r2_score`
- :func:`sklearn.metrics.zero_one_loss`

الأدوات
-------

- :func:`model_selection.train_test_split`

من المتوقع أن تنمو التغطية بمرور الوقت. يرجى اتباع `مشكلة التعريف على GitHub <https://github.com/scikit-learn/scikit-learn/issues/22352>`_ لتتبع التقدم.

نوع قيم الإرجاع والسمات المناسبة
-------------------------------------------

عند استدعاء الدوال أو الطرق مع مدخلات متوافقة مع واجهة برمجة تطبيقات المصفوفة، فإن الاصطلاح هو إرجاع قيم المصفوفة من نفس نوع حاوية المصفوفة والجهاز مثل بيانات الإدخال.

وبالمثل، عندما يتم ملاءمة مقدر مع مدخلات متوافقة مع واجهة برمجة تطبيقات المصفوفة، ستكون السمات المناسبة عبارة عن مصفوفات من نفس المكتبة مثل الإدخال ويتم تخزينها على نفس الجهاز.
تتوقع طريقة `predict` و `transform` لاحقًا مدخلات من نفس مكتبة المصفوفة والجهاز مثل البيانات التي تم تمريرها إلى طريقة `fit`.

لاحظ مع ذلك أن وظائف التسجيل التي تُرجع قيمًا عددية تُرجع عدديًا Python (عادةً مثيل `float`) بدلاً من قيمة عددية للمصفوفة.

فحوصات المقدر الشائعة
=======================

أضف علامة `array_api_support` إلى مجموعة علامات المقدر للإشارة إلى أنه يدعم واجهة برمجة تطبيقات المصفوفة.
سيؤدي ذلك إلى تمكين عمليات فحص مخصصة كجزء من الاختبارات الشائعة للتحقق من أن نتائج المقدرات هي نفسها عند استخدام مدخلات NumPy و Array API العادية.

لتشغيل هذه الفحوصات، تحتاج إلى تثبيت `array_api_compat <https://github.com/data-apis/array-api-compat>`_ في بيئة الاختبار الخاصة بك.
لتشغيل مجموعة الفحوصات الكاملة، تحتاج إلى تثبيت كل من `PyTorch <https://pytorch.org/>`_ و `CuPy <https://cupy.dev/>`_ ولديك وحدة معالجة رسومات.
سيتم تخطي عمليات الفحص التي لا يمكن تنفيذها أو التي تفتقد إلى تبعيات تلقائيًا.
لذلك من المهم تشغيل الاختبارات باستخدام علامة `-v` لمعرفة عمليات الفحص التي تم تخطيها:

.. prompt:: bash $

    pip install array-api-compat  # والمكتبات الأخرى حسب الحاجة
    pytest -k "array_api" -v

.. _mps_support:

ملاحظة حول دعم جهاز MPS
--------------------------

على macOS، يمكن لـ PyTorch استخدام Metal Performance Shaders (MPS) للوصول إلى مسرعات الأجهزة (على سبيل المثال، مكون وحدة معالجة الرسومات الداخلية لرقائق M1 أو M2).
ومع ذلك، فإن دعم جهاز MPS لـ PyTorch غير مكتمل في وقت كتابة هذا التقرير. راجع مشكلة github التالية لمزيد من التفاصيل:

- https://github.com/pytorch/pytorch/issues/77764

لتمكين دعم MPS في PyTorch، قم بتعيين متغير البيئة `PYTORCH_ENABLE_MPS_FALLBACK=1` قبل تشغيل الاختبارات:

.. prompt:: bash $

    PYTORCH_ENABLE_MPS_FALLBACK=1 pytest -k "array_api" -v

في وقت كتابة هذا التقرير، يجب أن تجتاز جميع اختبارات scikit-learn، ومع ذلك، فإن سرعة الحساب ليست بالضرورة أفضل من سرعة جهاز وحدة المعالجة المركزية.

.. _device_support_for_float64:

ملاحظة حول دعم الجهاز لـ ``float64``
--------------------------------------

ستؤدي عمليات معينة داخل scikit-learn تلقائيًا إلى إجراء عمليات على قيم الفاصلة العائمة بدقة `float64` لمنع الفائض وضمان الصحة (على سبيل المثال، :func:`metrics.pairwise.euclidean_distances`).
ومع ذلك، فإن مجموعات معينة من مساحات أسماء المصفوفة والأجهزة، مثل `PyTorch on MPS` (انظر :ref:`mps_support`) لا تدعم نوع البيانات `float64`.
في هذه الحالات، سيعود scikit-learn إلى استخدام نوع البيانات `float32` بدلاً من ذلك.
يمكن أن يؤدي ذلك إلى سلوك مختلف (عادةً نتائج غير مستقرة عدديًا) مقارنة بعدم استخدام إرسال واجهة برمجة تطبيقات المصفوفة أو استخدام جهاز يدعم `float64`.



