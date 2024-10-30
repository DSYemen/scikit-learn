.. currentmodule:: sklearn

.. TODO: update doc/conftest.py once document is updated and examples run.

.. _metadata_routing:

توجيه (إدارة) البيانات الوصفية
=================================

.. note::
  واجهة برمجة تطبيقات توجيه البيانات الوصفية تجريبية، ولم يتم تنفيذها بعد لجميع المقدرات.
  يرجى الرجوع إلى :ref:`قائمة النماذج المدعومة وغير المدعومة <metadata_routing_models>` لمزيد من المعلومات.
  قد تتغير بدون دورة الإهلاك المعتادة. افتراضيًا، هذه الميزة غير ممكّنة.
  يمكنك تمكينه عن طريق تعيين علامة ``enable_metadata_routing`` إلى ``True``::

    >>> import sklearn
    >>> sklearn.set_config(enable_metadata_routing=True)

  لاحظ أن الطرق والمتطلبات المقدمة في هذه الوثيقة ذات صلة فقط إذا كنت تريد تمرير :term:`metadata` (على سبيل المثال ``sample_weight``) إلى طريقة.
  إذا كنت تمرر فقط ``X`` و ``y`` ولا توجد معلمة / بيانات وصفية أخرى إلى طرق مثل :term:`fit`، :term:`transform`، وما إلى ذلك، فلن تحتاج إلى تعيين أي شيء.

يوضح هذا الدليل كيف يمكن توجيه :term:`metadata` وتمريرها بين الكائنات في scikit-learn.
إذا كنت تقوم بتطوير مقدر أو مقدر وصفية متوافق مع scikit-learn، فيمكنك مراجعة دليل المطور ذي الصلة:
:ref:`sphx_glr_auto_examples_miscellaneous_plot_metadata_routing.py`.

البيانات الوصفية هي بيانات يأخذها المقدر أو المسجل أو مقسم CV في الاعتبار إذا قام المستخدم بتمريرها صراحةً كمعلمة.
على سبيل المثال، يقبل :class:`~cluster.KMeans` `sample_weight` في طريقة `fit()` ويعتبرها لحساب مراكزها.
يتم استهلاك `classes` بواسطة بعض المصنفات ويتم استخدام `groups` في بعض الموصلات، ولكن يمكن اعتبار أي بيانات يتم تمريرها إلى طرق كائن بصرف النظر عن X و y كبيانات وصفية.
قبل إصدار scikit-learn 1.3، لم تكن هناك واجهة برمجة تطبيقات واحدة لتمرير البيانات الوصفية مثل ذلك إذا تم استخدام هذه الكائنات جنبًا إلى جنب مع كائنات أخرى، على سبيل المثال مسجل يقبل `sample_weight` داخل :class:`~model_selection.GridSearchCV`.

مع Metadata Routing API، يمكننا نقل البيانات الوصفية إلى المقدرات والمسجلين ومقسمات CV باستخدام :term:`meta-estimators` (مثل :class:`~pipeline.Pipeline` أو :class:`~model_selection.GridSearchCV`) أو وظائف مثل :func:`~model_selection.cross_validate` التي توجه البيانات إلى كائنات أخرى.
من أجل تمرير البيانات الوصفية إلى طريقة مثل ``fit`` أو ``score``، يجب أن *يطلبها* الكائن الذي يستهلك البيانات الوصفية.
يتم ذلك عبر طرق `set_{method}_request()`، حيث يتم استبدال `{method}` باسم الطريقة التي تطلب البيانات الوصفية.
على سبيل المثال، ستستخدم المقدرات التي تستخدم البيانات الوصفية في طريقة `fit()` `set_fit_request()`، وسيستخدم المسجلون `set_score_request()`.
تسمح لنا هذه الطرق بتحديد البيانات الوصفية المطلوبة، على سبيل المثال `set_fit_request(sample_weight=True)`.

بالنسبة للموصلات المجمعة مثل :class:`~model_selection.GroupKFold`، يتم طلب معلمة ``groups`` افتراضيًا.
من الأفضل توضيح ذلك من خلال الأمثلة التالية.

أمثلة الاستخدام
****************
نقدم هنا بعض الأمثلة لإظهار بعض حالات الاستخدام الشائعة.
هدفنا هو تمرير `sample_weight` و `groups` من خلال :func:`~model_selection.cross_validate`، التي توجه البيانات الوصفية إلى :class:`~linear_model.LogisticRegressionCV` وإلى مسجل مخصص تم إجراؤه باستخدام :func:`~metrics.make_scorer`، وكلاهما *يمكنه* استخدام البيانات الوصفية في طرقهما.
في هذه الأمثلة، نريد تعيين ما إذا كان سيتم استخدام البيانات الوصفية بشكل فردي داخل :term:`consumers <consumer>` المختلفة.

تتطلب الأمثلة في هذا القسم الواردات والبيانات التالية::

  >>> import numpy as np
  >>> from sklearn.metrics import make_scorer, accuracy_score
  >>> from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
  >>> from sklearn.model_selection import cross_validate, GridSearchCV, GroupKFold
  >>> from sklearn.feature_selection import SelectKBest
  >>> from sklearn.pipeline import make_pipeline
  >>> n_samples, n_features = 100, 4
  >>> rng = np.random.RandomState(42)
  >>> X = rng.rand(n_samples, n_features)
  >>> y = rng.randint(0, 2, size=n_samples)
  >>> my_groups = rng.randint(0, 10, size=n_samples)
  >>> my_weights = rng.rand(n_samples)
  >>> my_other_weights = rng.rand(n_samples)

التسجيل المرجح والملاءمة
----------------------------

يطلب الموصل المستخدم داخليًا في :class:`~linear_model.LogisticRegressionCV`، :class:`~model_selection.GroupKFold`، ``groups`` افتراضيًا.
ومع ذلك، نحتاج إلى طلب `sample_weight` صراحةً له وللمسجل المخصص لدينا عن طريق تحديد `sample_weight=True` في طريقة :class:`~linear_model.LogisticRegressionCV`s `set_fit_request()` وفي طريقة :func:`~metrics.make_scorer`s `set_score_request()`.
يعرف كل من :term:`consumers <consumer>` كيفية استخدام ``sample_weight`` في طرقهما `fit()` أو `score()`. يمكننا بعد ذلك تمرير البيانات الوصفية في :func:`~model_selection.cross_validate` التي ستوجهها إلى أي مستهلكين نشطين::

  >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
  >>> lr = LogisticRegressionCV(
  ...     cv=GroupKFold(),
  ...     scoring=weighted_acc
  ... ).set_fit_request(sample_weight=True)
  >>> cv_results = cross_validate(
  ...     lr,
  ...     X,
  ...     y,
  ...     params={"sample_weight": my_weights, "groups": my_groups},
  ...     cv=GroupKFold(),
  ...     scoring=weighted_acc,
  ... )

لاحظ أنه في هذا المثال، تقوم :func:`~model_selection.cross_validate` بتوجيه ``my_weights`` إلى كل من المسجل و :class:`~linear_model.LogisticRegressionCV`.

إذا مررنا `sample_weight` في معلمات :func:`~model_selection.cross_validate`، ولكن لم نقم بتعيين أي كائن لطلبه، فسيتم رفع `UnsetMetadataPassedError`، مما يشير إلينا إلى أننا نحتاج إلى تعيين مكان توجيهه صراحةً.
ينطبق الشيء نفسه إذا تم تمرير ``params={"sample_weights": my_weights، ...}`` (لاحظ الخطأ المطبعي، أي ``weights`` بدلاً من ``weight``)، نظرًا لأن ``sample_weights`` لم يتم طلبها بواسطة أي من الكائنات الأساسية.

التسجيل المرجح والملاءمة غير المرجحة
---------------------------------------

عند تمرير البيانات الوصفية مثل ``sample_weight`` إلى :term:`router` (:term:`meta-estimators` أو دالة التوجيه)، تتطلب جميع :term:`consumers <consumer>` ``sample_weight`` أن يتم طلب الأوزان صراحةً أو عدم طلبها صراحةً (على سبيل المثال ``True`` أو ``False``).
وبالتالي، لإجراء ملاءمة غير مرجحة، نحتاج إلى تكوين :class:`~linear_model.LogisticRegressionCV` لعدم طلب أوزان العينة، بحيث لا تمرر :func:`~model_selection.cross_validate` الأوزان على طول::

  >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
  >>> lr = LogisticRegressionCV(
  ...     cv=GroupKFold(), scoring=weighted_acc,
  ... ).set_fit_request(sample_weight=False)
  >>> cv_results = cross_validate(
  ...     lr,
  ...     X,
  ...     y,
  ...     cv=GroupKFold(),
  ...     params={"sample_weight": my_weights, "groups": my_groups},
  ...     scoring=weighted_acc,
  ... )

إذا لم يتم استدعاء :meth:`linear_model.LogisticRegressionCV.set_fit_request`، فستثير :func:`~model_selection.cross_validate` خطأً لأن ``sample_weight`` تم تمريره ولكن :class:`~linear_model.LogisticRegressionCV` لن يتم تكوينه صراحةً للتعرف على الأوزان.

اختيار الميزة غير المرجحة
----------------------------

لا يكون توجيه البيانات الوصفية ممكنًا إلا إذا كانت طريقة الكائن تعرف كيفية استخدام البيانات الوصفية، مما يعني في معظم الحالات أنها تحتوي عليها كمعلمة صريحة.
عندها فقط يمكننا تعيين قيم الطلب للبيانات الوصفية باستخدام `set_fit_request(sample_weight=True)`، على سبيل المثال.
هذا يجعل الكائن :term:`consumer <consumer>`.

على عكس :class:`~linear_model.LogisticRegressionCV`، لا يمكن لـ :class:`~feature_selection.SelectKBest` استهلاك الأوزان وبالتالي لا يتم تعيين قيمة طلب لـ ``sample_weight`` على مثيله ولا يتم توجيه ``sample_weight`` إليه::

  >>> weighted_acc = make_scorer(accuracy_score).set_score_request(sample_weight=True)
  >>> lr = LogisticRegressionCV(
  ...     cv=GroupKFold(), scoring=weighted_acc,
  ... ).set_fit_request(sample_weight=True)
  >>> sel = SelectKBest(k=2)
  >>> pipe = make_pipeline(sel, lr)
  >>> cv_results = cross_validate(
  ...     pipe,
  ...     X,
  ...     y,
  ...     cv=GroupKFold(),
  ...     params={"sample_weight": my_weights, "groups": my_groups},
  ...     scoring=weighted_acc,
  ... )

أوزان تسجيل وملاءمة مختلفة
--------------------------------

على الرغم من أن كل من :func:`~metrics.make_scorer` و :class:`~linear_model.LogisticRegressionCV` يتوقعان المفتاح ``sample_weight``، يمكننا استخدام الأسماء المستعارة لتمرير أوزان مختلفة إلى مستهلكين مختلفين.
في هذا المثال، نمرر ``scoring_weight`` إلى المسجل، و ``fitting_weight`` إلى :class:`~linear_model.LogisticRegressionCV`::

  >>> weighted_acc = make_scorer(accuracy_score).set_score_request(
  ...    sample_weight="scoring_weight"
  ... )
  >>> lr = LogisticRegressionCV(
  ...     cv=GroupKFold(), scoring=weighted_acc,
  ... ).set_fit_request(sample_weight="fitting_weight")
  >>> cv_results = cross_validate(
  ...     lr,
  ...     X,
  ...     y,
  ...     cv=GroupKFold(),
  ...     params={
  ...         "scoring_weight": my_weights,
  ...         "fitting_weight": my_other_weights,
  ...         "groups": my_groups,
  ...     },
  ...     scoring=weighted_acc,
  ... )

واجهة API
*************

:term:`consumer` هو كائن (مقدر، مقدر وصفية، مسجل، مقسم) يقبل ويستخدم بعض :term:`metadata` في طريقة واحدة على الأقل من طرقه (على سبيل المثال ``fit``، ``predict``، ``inverse_transform``، ``transform``، ``score``، ``split``).
المقدرات الوصفية التي تقوم فقط بإعادة توجيه البيانات الوصفية إلى كائنات أخرى (مقدرات تابعة، أو مسجلون، أو مقسمات) ولا تستخدم البيانات الوصفية نفسها ليست مستهلكين.
(Meta-) المقدرات التي توجه البيانات الوصفية إلى كائنات أخرى هي :term:`routers <router>`.
يمكن أن يكون (n) (meta-) المقدر :term:`consumer` و :term:`router` في نفس الوقت.
تعرض (Meta-) المقدرات والموصلات طريقة `set_{method}_request` لكل طريقة تقبل بيانات وصفية واحدة على الأقل.
على سبيل المثال، إذا كان المقدر يدعم ``sample_weight`` في ``fit`` و ``score``، فإنه يعرض ``estimator.set_fit_request(sample_weight=value)`` و ``estimator.set_score_request(sample_weight=value)``.
هنا يمكن أن تكون ``value``:

- ``True``: تطلب الطريقة ``sample_weight``. هذا يعني أنه إذا تم توفير البيانات الوصفية، فسيتم استخدامها، وإلا فلن يتم رفع أي خطأ.
- ``False``: لا تطلب الطريقة ``sample_weight``.
- ``None``: سيرفع جهاز التوجيه خطأً إذا تم تمرير ``sample_weight``.
  هذه هي القيمة الافتراضية في جميع الحالات تقريبًا عند إنشاء مثيل لكائن وتضمن أن يقوم المستخدم بتعيين طلبات البيانات الوصفية صراحةً عند تمرير البيانات الوصفية.
  الاستثناء الوحيد هو مقسمات ``Group*Fold``.
- ``"param_name"``: اسم مستعار لـ ``sample_weight`` إذا أردنا تمرير أوزان مختلفة إلى مستهلكين مختلفين.
  إذا تم استخدام الاسم المستعار، فلا ينبغي أن يقوم المقدر الوصفي بإعادة توجيه ``"param_name"`` إلى المستهلك، ولكن ``sample_weight`` بدلاً من ذلك، لأن المستهلك سيتوقع معلمة تسمى ``sample_weight``.
  هذا يعني أن التعيين بين البيانات الوصفية التي يتطلبها الكائن، على سبيل المثال ``sample_weight`` واسم المتغير الذي يوفره المستخدم، على سبيل المثال ``my_weights`` يتم على مستوى جهاز التوجيه، وليس بواسطة الكائن المستهلك نفسه.

يتم طلب البيانات الوصفية بنفس الطريقة للمسجلين باستخدام ``set_score_request``.

إذا تم تمرير بيانات وصفية، على سبيل المثال ``sample_weight``، بواسطة المستخدم، فيجب على المستخدم تعيين طلب البيانات الوصفية لجميع الكائنات التي يمكن أن تستهلك ``sample_weight``، وإلا فسيتم رفع خطأ بواسطة كائن جهاز التوجيه.
على سبيل المثال، يثير الكود التالي خطأً، لأنه لم يتم تحديد ما إذا كان يجب تمرير ``sample_weight`` إلى مسجل المقدر أم لا::

    >>> param_grid = {"C": [0.1, 1]}
    >>> lr = LogisticRegression().set_fit_request(sample_weight=True)
    >>> try:
    ...     GridSearchCV(
    ...         estimator=lr, param_grid=param_grid
    ...     ).fit(X, y, sample_weight=my_weights)
    ... except ValueError as e:
    ...     print(e)
    [sample_weight] تم تمريرها ولكن لم يتم تعيينها صراحةً على أنها مطلوبة أو غير مطلوبة لـ LogisticRegression.score، والتي يتم استخدامها داخل GridSearchCV.fit.
    اتصل بـ `LogisticRegression.set_score_request({metadata}=True/False)` لكل بيانات وصفية
    تريد طلبها / تجاهلها.

يمكن إصلاح المشكلة عن طريق تعيين قيمة الطلب صراحةً::

    >>> lr = LogisticRegression().set_fit_request(
    ...     sample_weight=True
    ... ).set_score_request(sample_weight=False)

في نهاية قسم **أمثلة الاستخدام**، نقوم بتعطيل علامة التكوين لتوجيه البيانات الوصفية::

    >>> sklearn.set_config(enable_metadata_routing=False)


.. _metadata_routing_models:

حالة دعم توجيه البيانات الوصفية
***********************************
يدعم جميع المستهلكين (أي المقدرات البسيطة التي تستهلك البيانات الوصفية فقط ولا توجهها) توجيه البيانات الوصفية، مما يعني أنه يمكن استخدامها داخل المقدرات الوصفية التي تدعم توجيه البيانات الوصفية.
ومع ذلك، فإن تطوير دعم توجيه البيانات الوصفية للمقدرات الوصفية قيد التقدم، وهنا قائمة بالمقدرات الوصفية والأدوات التي تدعم ولا تدعم توجيه البيانات الوصفية بعد.


المقدرات الوصفية والوظائف التي تدعم توجيه البيانات الوصفية:

- :class:`sklearn.calibration.CalibratedClassifierCV`
- :class:`sklearn.compose.ColumnTransformer`
- :class:`sklearn.compose.TransformedTargetRegressor`
- :class:`sklearn.covariance.GraphicalLassoCV`
- :class:`sklearn.ensemble.StackingClassifier`
- :class:`sklearn.ensemble.StackingRegressor`
- :class:`sklearn.ensemble.VotingClassifier`
- :class:`sklearn.ensemble.VotingRegressor`
- :class:`sklearn.ensemble.BaggingClassifier`
- :class:`sklearn.ensemble.BaggingRegressor`
- :class:`sklearn.feature_selection.RFE`
- :class:`sklearn.feature_selection.RFECV`
- :class:`sklearn.feature_selection.SelectFromModel`
- :class:`sklearn.feature_selection.SequentialFeatureSelector`
- :class:`sklearn.impute.IterativeImputer`
- :class:`sklearn.linear_model.ElasticNetCV`
- :class:`sklearn.linear_model.LarsCV`
- :class:`sklearn.linear_model.LassoCV`
- :class:`sklearn.linear_model.LassoLarsCV`
- :class:`sklearn.linear_model.LogisticRegressionCV`
- :class:`sklearn.linear_model.MultiTaskElasticNetCV`
- :class:`sklearn.linear_model.MultiTaskLassoCV`
- :class:`sklearn.linear_model.OrthogonalMatchingPursuitCV`
- :class:`sklearn.linear_model.RANSACRegressor`
- :class:`sklearn.linear_model.RidgeClassifierCV`
- :class:`sklearn.linear_model.RidgeCV`
- :class:`sklearn.model_selection.GridSearchCV`
- :class:`sklearn.model_selection.HalvingGridSearchCV`
- :class:`sklearn.model_selection.HalvingRandomSearchCV`
- :class:`sklearn.model_selection.RandomizedSearchCV`
- :class:`sklearn.model_selection.permutation_test_score`
- :func:`sklearn.model_selection.cross_validate`
- :func:`sklearn.model_selection.cross_val_score`
- :func:`sklearn.model_selection.cross_val_predict`
- :class:`sklearn.model_selection.learning_curve`
- :class:`sklearn.model_selection.validation_curve`
- :class:`sklearn.multiclass.OneVsOneClassifier`
- :class:`sklearn.multiclass.OneVsRestClassifier`
- :class:`sklearn.multiclass.OutputCodeClassifier`
- :class:`sklearn.multioutput.ClassifierChain`
- :class:`sklearn.multioutput.MultiOutputClassifier`
- :class:`sklearn.multioutput.MultiOutputRegressor`
- :class:`sklearn.multioutput.RegressorChain`
- :class:`sklearn.pipeline.FeatureUnion`
- :class:`sklearn.pipeline.Pipeline`
- :class:`sklearn.semi_supervised.SelfTrainingClassifier`

المقدرات الوصفية والأدوات التي لا تدعم توجيه البيانات الوصفية بعد:

- :class:`sklearn.ensemble.AdaBoostClassifier`
- :class:`sklearn.ensemble.AdaBoostRegressor`



