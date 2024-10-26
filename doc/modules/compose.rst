
.. _combining_estimators:

==================================
خطوط الأنابيب والمقدرات المركبة
==================================

لبناء مقدر مركب، عادةً ما يتم دمج المحولات مع محولات أخرى أو مع :term:`المتنبئين` (مثل المصنفات أو عوامل الانحدار).
الأداة الأكثر شيوعًا المستخدمة لتكوين المقدرات هي :ref:`Pipeline <pipeline>`.
تتطلب خطوط الأنابيب أن تكون جميع الخطوات باستثناء الأخيرة :term:`محول`.
يمكن أن تكون الخطوة الأخيرة أي شيء، محول، :term:`متنبئ`، أو مقدر تجميع قد يكون أو لا يحتوي على طريقة `.predict(...)`.
يعرض خط الأنابيب جميع الطرق التي يوفرها المقدر الأخير: إذا كانت الخطوة الأخيرة توفر طريقة `transform`، فسيكون لخط الأنابيب طريقة `transform` ويتصرف مثل المحول.
إذا كانت الخطوة الأخيرة توفر طريقة `predict`، فسيكشف خط الأنابيب عن تلك الطريقة، وبالنظر إلى البيانات :term:`X`، استخدم جميع الخطوات باستثناء الأخيرة لتحويل البيانات، ثم أعط تلك البيانات المحولة إلى طريقة `predict` للخطوة الأخيرة من خط الأنابيب.
غالبًا ما تُستخدم الفئة :class:`Pipeline` مع :ref:`ColumnTransformer <column_transformer>` أو :ref:`FeatureUnion <feature_union>` التي تربط ناتج المحولات في مساحة ميزة مركبة.
يتعامل :ref:`TransformedTargetRegressor <transformed_target_regressor>` مع تحويل :term:`الهدف` (أي تحويل السجل :term:`y`).

.. _pipeline:

Pipeline: سلسلة المقدرات
=============================

.. currentmodule:: sklearn.pipeline

يمكن استخدام :class:`Pipeline` لسلسلة مقدرات متعددة في واحد.
هذا مفيد لأنه غالبًا ما يكون هناك تسلسل ثابت للخطوات في معالجة البيانات، على سبيل المثال اختيار الميزات والتطبيع والتصنيف. يخدم :class:`Pipeline` أغراضًا متعددة هنا:

- الراحة والتغليف
    ما عليك سوى استدعاء :term:`fit` و :term:`predict` مرة واحدة على بياناتك لتناسب تسلسل كامل من المقدرات.
- اختيار المعلمة المشتركة
    يمكنك :ref:`البحث في الشبكة <grid_search>` على معلمات جميع المقدرات في خط الأنابيب في وقت واحد.
- الأمان
    تساعد خطوط الأنابيب على تجنب تسريب الإحصائيات من بيانات الاختبار الخاصة بك إلى النموذج المدرب في التحقق المتبادل، من خلال ضمان استخدام نفس العينات لتدريب المحولات والمتنبئين.

يجب أن تكون جميع المقدرات في خط الأنابيب، باستثناء الأخير، محولات (أي يجب أن يكون لها طريقة :term:`transform`).
قد يكون المقدر الأخير من أي نوع (محول، مصنف، إلخ).

.. note::

    إن استدعاء ``fit`` على خط الأنابيب هو نفسه استدعاء ``fit`` على كل مقدر بدوره، ``transform``  الإدخال وتمريره إلى الخطوة التالية.
    يحتوي خط الأنابيب على جميع الطرق التي يمتلكها المقدر الأخير في خط الأنابيب، أي إذا كان المقدر الأخير هو مصنف، فيمكن استخدام :class:`Pipeline`  كمصنف.
    إذا كان المقدر الأخير هو محول، مرة أخرى، كذلك هو خط الأنابيب.


الاستخدام
---------

بناء خط أنابيب
................

يتم بناء :class:`Pipeline` باستخدام قائمة من أزواج ``(key, value)``، حيث ``key`` عبارة عن سلسلة تحتوي على الاسم الذي تريد إعطائه لهذه الخطوة و ``value`` هو كائن مقدر::

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.svm import SVC
    >>> from sklearn.decomposition import PCA
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> pipe = Pipeline(estimators)
    >>> pipe
    Pipeline(steps=[('reduce_dim', PCA()), ('clf', SVC())])

.. dropdown:: إصدار مختصر باستخدام :func:`make_pipeline`

  دالة الأداة المساعدة :func:`make_pipeline` هي اختصار لبناء خطوط الأنابيب؛ يأخذ عددًا متغيرًا من المقدرات ويعيد خط أنابيب، مع ملء الأسماء تلقائيًا::

      >>> from sklearn.pipeline import make_pipeline
      >>> make_pipeline(PCA(), SVC())
      Pipeline(steps=[('pca', PCA()), ('svc', SVC())])

الوصول إلى خطوات خط الأنابيب
...............................

يتم تخزين مقدرات خط الأنابيب كقائمة في سمة ``steps``.
يمكن استخراج خط أنابيب فرعي باستخدام تدوين التقطيع المستخدم بشكل شائع لتسلسلات Python مثل القوائم أو السلاسل (على الرغم من أنه يُسمح بخطوة واحدة فقط).
هذا مناسب لأداء بعض التحويلات فقط (أو عكسها)::

    >>> pipe[:1]
    Pipeline(steps=[('reduce_dim', PCA())])
    >>> pipe[-1:]
    Pipeline(steps=[('clf', SVC())])

.. dropdown:: الوصول إلى خطوة بالاسم أو الموضع

  يمكن أيضًا الوصول إلى خطوة محددة عن طريق الفهرس أو الاسم عن طريق فهرسة (مع ``[idx]``) خط الأنابيب::

      >>> pipe.steps[0]
      ('reduce_dim', PCA())
      >>> pipe[0]
      PCA()
      >>> pipe['reduce_dim']
      PCA()

  تسمح سمة `named_steps` لـ `Pipeline` بالوصول إلى الخطوات بالاسم مع إكمال علامة التبويب في البيئات التفاعلية::

      >>> pipe.named_steps.reduce_dim is pipe['reduce_dim']
      True

تتبع أسماء الميزات في خط أنابيب
....................................

لتمكين فحص النموذج، :class:`~sklearn.pipeline.Pipeline` لديه طريقة ``get_feature_names_out()``، تمامًا مثل جميع المحولات.
يمكنك استخدام تقطيع خط الأنابيب للحصول على أسماء الميزات التي تدخل في كل خطوة::

    >>> from sklearn.datasets import load_iris
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.feature_selection import SelectKBest
    >>> iris = load_iris()
    >>> pipe = Pipeline(steps=[
    ...    ('select', SelectKBest(k=2)),
    ...    ('clf', LogisticRegression())])
    >>> pipe.fit(iris.data, iris.target)
    Pipeline(steps=[('select', SelectKBest(...)), ('clf', LogisticRegression(...))])
    >>> pipe[:-1].get_feature_names_out()
    array(['x2', 'x3'], ...)

.. dropdown:: تخصيص أسماء الميزات

  يمكنك أيضًا توفير أسماء ميزات مخصصة لبيانات الإدخال باستخدام ``get_feature_names_out``::

      >>> pipe[:-1].get_feature_names_out(iris.feature_names)
      array(['petal length (cm)', 'petal width (cm)'], ...)

.. _pipeline_nested_parameters:

الوصول إلى المعلمات المتداخلة
.................................

من الشائع تعديل معلمات المقدر داخل خط الأنابيب.
لذلك يتم تداخل هذه المعلمة لأنها تنتمي إلى خطوة فرعية معينة.
يمكن الوصول إلى معلمات المقدرات في خط الأنابيب باستخدام بناء جملة ``<estimator>__<parameter>``::

    >>> pipe = Pipeline(steps=[("reduce_dim", PCA()), ("clf", SVC())])
    >>> pipe.set_params(clf__C=10)
    Pipeline(steps=[('reduce_dim', PCA()), ('clf', SVC(C=10))])

.. dropdown:: متى يهم؟

  هذا مهم بشكل خاص لإجراء عمليات بحث الشبكة::

      >>> from sklearn.model_selection import GridSearchCV
      >>> param_grid = dict(reduce_dim__n_components=[2, 5, 10],
      ...                   clf__C=[0.1, 10, 100])
      >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)

  يمكن أيضًا استبدال الخطوات الفردية كمعلمات، ويمكن تجاهل الخطوات غير النهائية عن طريق تعيينها على ``'passthrough'``::

      >>> param_grid = dict(reduce_dim=['passthrough', PCA(5), PCA(10)],
      ...                   clf=[SVC(), LogisticRegression()],
      ...                   clf__C=[0.1, 10, 100])
      >>> grid_search = GridSearchCV(pipe, param_grid=param_grid)

  .. seealso::

    * :ref:`composite_grid_search`


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection/plot_feature_selection_pipeline.py`
* :ref:`sphx_glr_auto_examples_model_selection/plot_grid_search_text_feature_extraction.py`
* :ref:`sphx_glr_auto_examples_compose/plot_digits_pipe.py`
* :ref:`sphx_glr_auto_examples_miscellaneous/plot_kernel_approximation.py`
* :ref:`sphx_glr_auto_examples_svm/plot_svm_anova.py`
* :ref:`sphx_glr_auto_examples_compose/plot_compare_reduction.py`
* :ref:`sphx_glr_auto_examples_miscellaneous/plot_pipeline_display.py`


.. _pipeline_cache:

تخزين المحولات مؤقتًا: تجنب الحساب المتكرر
-------------------------------------------------

.. currentmodule:: sklearn.pipeline

قد يكون تركيب المحولات مكلفًا من الناحية الحسابية. مع تعيين معلمة ``memory``، سيقوم :class:`Pipeline` بتخزين كل محول مؤقتًا بعد استدعاء ``fit``.
تُستخدم هذه الميزة لتجنب حساب المحولات المناسبة داخل خط الأنابيب إذا كانت المعلمات وبيانات الإدخال متطابقة.
مثال نموذجي هو حالة بحث الشبكة حيث يمكن تركيب المحولات مرة واحدة فقط وإعادة استخدامها لكل تكوين.
لن يتم تخزين الخطوة الأخيرة مؤقتًا أبدًا، حتى لو كانت محولًا.

المعلمة ``memory`` مطلوبة من أجل تخزين المحولات مؤقتًا.
يمكن أن يكون ``memory`` إما سلسلة تحتوي على الدليل حيث يتم تخزين المحولات مؤقتًا أو كائن `joblib.Memory <https://joblib.readthedocs.io/en/latest/memory.html>`_::

    >>> from tempfile import mkdtemp
    >>> from shutil import rmtree
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.svm import SVC
    >>> from sklearn.pipeline import Pipeline
    >>> estimators = [('reduce_dim', PCA()), ('clf', SVC())]
    >>> cachedir = mkdtemp()
    >>> pipe = Pipeline(estimators, memory=cachedir)
    >>> pipe
    Pipeline(memory=...,
             steps=[('reduce_dim', PCA()), ('clf', SVC())])
    >>> # امسح دليل التخزين المؤقت عندما لا تعود بحاجة إليه
    >>> rmtree(cachedir)

.. dropdown:: الآثار الجانبية لتخزين المحولات مؤقتًا
  :color: warning

  باستخدام :class:`Pipeline` بدون تمكين التخزين المؤقت، من الممكن فحص المثيل الأصلي مثل::

      >>> from sklearn.datasets import load_digits
      >>> X_digits, y_digits = load_digits(return_X_y=True)
      >>> pca1 = PCA(n_components=10)
      >>> svm1 = SVC()
      >>> pipe = Pipeline([('reduce_dim', pca1), ('clf', svm1)])
      >>> pipe.fit(X_digits, y_digits)
      Pipeline(steps=[('reduce_dim', PCA(n_components=10)), ('clf', SVC())])
      >>> # يمكن فحص مثيل pca مباشرة
      >>> pca1.components_.shape
      (10, 64)

  يؤدي تمكين التخزين المؤقت إلى استنساخ المحولات قبل التركيب.
  لذلك، لا يمكن فحص مثيل المحول المعطى لخط الأنابيب مباشرةً.
  في المثال التالي، سيؤدي الوصول إلى مثيل :class:`~sklearn.decomposition.PCA` ``pca2`` إلى حدوث ``AttributeError`` لأن ``pca2`` سيكون محولًا غير مناسب.
  بدلاً من ذلك، استخدم سمة ``named_steps`` لفحص المقدرات داخل خط الأنابيب::

      >>> cachedir = mkdtemp()
      >>> pca2 = PCA(n_components=10)
      >>> svm2 = SVC()
      >>> cached_pipe = Pipeline([('reduce_dim', pca2), ('clf', svm2)],
      ...                        memory=cachedir)
      >>> cached_pipe.fit(X_digits, y_digits)
      Pipeline(memory=...,
               steps=[('reduce_dim', PCA(n_components=10)), ('clf', SVC())])
      >>> cached_pipe.named_steps['reduce_dim'].components_.shape
      (10, 64)
      >>> # إزالة دليل التخزين المؤقت
      >>> rmtree(cachedir)


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_compose/plot_compare_reduction.py`

.. _transformed_target_regressor:

تحويل الهدف في الانحدار
=================================

يقوم :class:`~sklearn.compose.TransformedTargetRegressor` بتحويل الأهداف ``y`` قبل ملاءمة نموذج الانحدار.
يتم تعيين التنبؤات مرة أخرى إلى المساحة الأصلية عبر تحويل عكسي.
يأخذ كوسيطة عامل الانحدار الذي سيتم استخدامه للتنبؤ، والمحول الذي سيتم تطبيقه على متغير الهدف::

  >>> import numpy as np
  >>> from sklearn.datasets import fetch_california_housing
  >>> from sklearn.compose import TransformedTargetRegressor
  >>> from sklearn.preprocessing import QuantileTransformer
  >>> from sklearn.linear_model import LinearRegression
  >>> from sklearn.model_selection import train_test_split
  >>> X, y = fetch_california_housing(return_X_y=True)
  >>> X, y = X[:2000, :], y[:2000]  # حدد مجموعة فرعية من البيانات
  >>> transformer = QuantileTransformer(output_distribution='normal')
  >>> regressor = LinearRegression()
  >>> regr = TransformedTargetRegressor(regressor=regressor,
  ...                                   transformer=transformer)
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  >>> regr.fit(X_train, y_train)
  TransformedTargetRegressor(...)
  >>> print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
  R2 score: 0.61
  >>> raw_target_regr = LinearRegression().fit(X_train, y_train)
  >>> print('R2 score: {0:.2f}'.format(raw_target_regr.score(X_test, y_test)))
  R2 score: 0.59

بالنسبة للتحويلات البسيطة، بدلاً من كائن Transformer، زوج من
يمكن تمرير الدوال، وتحديد التحويل وتعيينه العكسي::

  >>> def func(x):
  ...     return np.log(x)
  >>> def inverse_func(x):
  ...     return np.exp(x)

بعد ذلك، يتم إنشاء الكائن على النحو التالي::

  >>> regr = TransformedTargetRegressor(regressor=regressor,
  ...                                   func=func,
  ...                                   inverse_func=inverse_func)
  >>> regr.fit(X_train, y_train)
  TransformedTargetRegressor(...)
  >>> print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
  R2 score: 0.51

افتراضيًا، يتم فحص الدوال المقدمة في كل ملاءمة لتكون معكوسة لبعضها البعض. ومع ذلك، من الممكن تجاوز هذا الفحص عن طريق تعيين ``check_inverse`` على ``False``::

  >>> def inverse_func(x):
  ...     return x
  >>> regr = TransformedTargetRegressor(regressor=regressor,
  ...                                   func=func,
  ...                                   inverse_func=inverse_func,
  ...                                   check_inverse=False)
  >>> regr.fit(X_train, y_train)
  TransformedTargetRegressor(...)
  >>> print('R2 score: {0:.2f}'.format(regr.score(X_test, y_test)))
  R2 score: -1.57

.. note::

   يمكن تشغيل التحويل عن طريق تعيين ``transformer`` أو الزوج
   من الدوال ``func`` و ``inverse_func``. ومع ذلك، فإن تعيين كلا الخيارين سيرفع خطأ.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_compose/plot_transformed_target.py`


.. _feature_union:

FeatureUnion: مساحات ميزات مركبة
======================================

.. currentmodule:: sklearn.pipeline

يقوم :class:`FeatureUnion` بدمج العديد من كائنات المحول في محول جديد يجمع مخرجاتها.
يأخذ :class:`FeatureUnion` قائمة بكائنات المحول.
أثناء التركيب، يتم ملاءمة كل من هذه البيانات بشكل مستقل.
يتم تطبيق المحولات بالتوازي، ويتم ربط مصفوفات الميزات التي تُخرجها جنبًا إلى جنب في مصفوفة أكبر.

عندما تريد تطبيق تحويلات مختلفة على كل حقل من البيانات، انظر الفئة ذات الصلة :class:`~sklearn.compose.ColumnTransformer` (انظر :ref:`دليل المستخدم <column_transformer>`).

يخدم :class:`FeatureUnion` نفس أغراض :class:`Pipeline` - الراحة وتقدير المعلمات والتحقق من الصحة المشتركة.

يمكن دمج :class:`FeatureUnion` و :class:`Pipeline` لإنشاء نماذج معقدة.

(لا توجد طريقة لـ :class:`FeatureUnion` للتحقق مما إذا كان محولين قد ينتجان ميزات متطابقة.
ينتج اتحادًا فقط عندما تكون مجموعات الميزات منفصلة، والتأكد من أنها مسؤولية المتصل.)


الاستخدام
--------

يتم بناء :class:`FeatureUnion` باستخدام قائمة من أزواج ``(key, value)``، حيث ``key`` هو الاسم الذي تريد إعطائه لتحويل معين (سلسلة عشوائية؛ فهو بمثابة معرف فقط) و ``value`` هو كائن مقدر::

    >>> from sklearn.pipeline import FeatureUnion
    >>> from sklearn.decomposition import PCA
    >>> from sklearn.decomposition import KernelPCA
    >>> estimators = [('linear_pca', PCA()), ('kernel_pca', KernelPCA())]
    >>> combined = FeatureUnion(estimators)
    >>> combined
    FeatureUnion(transformer_list=[('linear_pca', PCA()),
                                   ('kernel_pca', KernelPCA())])


مثل خطوط الأنابيب، تحتوي اتحادات الميزات على مُنشئ مختصر يسمى :func:`make_union` لا يتطلب تسمية صريحة للمكونات.


مثل ``Pipeline``، يمكن استبدال الخطوات الفردية باستخدام ``set_params``، وتجاهلها عن طريق التعيين على ``'drop'``::

    >>> combined.set_params(kernel_pca='drop')
    FeatureUnion(transformer_list=[('linear_pca', PCA()),
                                   ('kernel_pca', 'drop')])

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_compose/plot_feature_union.py`


.. _column_transformer:

ColumnTransformer للبيانات غير المتجانسة
========================================

تحتوي العديد من مجموعات البيانات على ميزات من أنواع مختلفة، على سبيل المثال النص والعوامات والتواريخ، حيث يتطلب كل نوع من الميزات خطوات معالجة مسبقة أو استخراج ميزات منفصلة.
غالبًا ما يكون من الأسهل معالجة البيانات مسبقًا قبل تطبيق طرق scikit-learn، على سبيل المثال باستخدام `pandas <https://pandas.pydata.org/>`__.
قد تكون معالجة بياناتك قبل تمريرها إلى scikit-learn مشكلة لأحد الأسباب التالية:

1. دمج الإحصائيات من بيانات الاختبار في المعالجات المسبقة يجعل درجات التحقق المتبادل غير موثوقة (تُعرف باسم *تسرب البيانات*)، على سبيل المثال في حالة المقاييس أو إدخال القيم المفقودة.
2. قد ترغب في تضمين معلمات المعالجات المسبقة في :ref:`بحث المعلمات <grid_search>`.

يساعد :class:`~sklearn.compose.ColumnTransformer` في إجراء تحويلات مختلفة لأعمدة مختلفة من البيانات، داخل :class:`~sklearn.pipeline.Pipeline` آمن من تسرب البيانات ويمكن تحديد معلمات له.
يعمل :class:`~sklearn.compose.ColumnTransformer` على المصفوفات والمصفوفات المتفرقة و `pandas DataFrames <https://pandas.pydata.org/pandas-docs/stable/>`__.

يمكن تطبيق تحويل مختلف على كل عمود، مثل المعالجة المسبقة أو طريقة استخراج ميزات محددة::

  >>> import pandas as pd
  >>> X = pd.DataFrame(
  ...     {'city': ['London', 'London', 'Paris', 'Sallisaw'],
  ...      'title': ["His Last Bow", "How Watson Learned the Trick",
  ...                "A Moveable Feast", "The Grapes of Wrath"],
  ...      'expert_rating': [5, 3, 4, 5],
  ...      'user_rating': [4, 5, 4, 3]})

بالنسبة لهذه البيانات، قد نرغب في ترميز عمود ``'city'`` كمتغير فئوي باستخدام :class:`~sklearn.preprocessing.OneHotEncoder` ولكن تطبيق :class:`~sklearn.feature_extraction.text.CountVectorizer` على عمود ``'title'``.
نظرًا لأننا قد نستخدم طرق استخراج ميزات متعددة على نفس العمود، فإننا نعطي كل محول اسمًا فريدًا، على سبيل المثال ``'city_category'`` و ``'title_bow'``.
افتراضيًا، يتم تجاهل أعمدة التصنيف المتبقية (``remainder='drop'``)::

  >>> from sklearn.compose import ColumnTransformer
  >>> from sklearn.feature_extraction.text import CountVectorizer
  >>> from sklearn.preprocessing import OneHotEncoder
  >>> column_trans = ColumnTransformer(
  ...     [('categories', OneHotEncoder(dtype='int'), ['city']),
  ...      ('title_bow', CountVectorizer(), 'title')],
  ...     remainder='drop', verbose_feature_names_out=False)

  >>> column_trans.fit(X)
  ColumnTransformer(transformers=[('categories', OneHotEncoder(dtype='int'),
                                   ['city']),
                                  ('title_bow', CountVectorizer(), 'title')],
                    verbose_feature_names_out=False)

  >>> column_trans.get_feature_names_out()
  array(['city_London', 'city_Paris', 'city_Sallisaw', 'bow', 'feast',
  'grapes', 'his', 'how', 'last', 'learned', 'moveable', 'of', 'the',
   'trick', 'watson', 'wrath'], ...)

  >>> column_trans.transform(X).toarray()
  array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
         [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0],
         [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1]]...)

في المثال أعلاه، يتوقع :class:`~sklearn.feature_extraction.text.CountVectorizer` مصفوفة أحادية البعد كمدخل، وبالتالي تم تحديد الأعمدة كسلسلة (``'title'``).
ومع ذلك، يتوقع :class:`~sklearn.preprocessing.OneHotEncoder` مثل معظم المحولات الأخرى بيانات ثنائية الأبعاد، لذلك في هذه الحالة تحتاج إلى تحديد العمود كقائمة من السلاسل (``['city']``).

بصرف النظر عن العددية أو قائمة العناصر الفردية، يمكن تحديد اختيار العمود كقائمة من عناصر متعددة، أو مصفوفة عدد صحيح، أو شريحة، أو قناع منطقي، أو باستخدام :func:`~sklearn.compose.make_column_selector`.
يتم استخدام :func:`~sklearn.compose.make_column_selector` لتحديد الأعمدة بناءً على نوع البيانات أو اسم العمود::

  >>> from sklearn.preprocessing import StandardScaler
  >>> from sklearn.compose import make_column_selector
  >>> ct = ColumnTransformer([
  ...       ('scale', StandardScaler(),
  ...       make_column_selector(dtype_include=np.number)),
  ...       ('onehot',
  ...       OneHotEncoder(),
  ...       make_column_selector(pattern='city', dtype_include=object))])
  >>> ct.fit_transform(X)
  array([[ 0.904...,  0.      ,  1. ,  0. ,  0. ],
         [-1.507...,  1.414...,  1. ,  0. ,  0. ],
         [-0.301...,  0.      ,  0. ,  1. ,  0. ],
         [ 0.904..., -1.414...,  0. ,  0. ,  1. ]])

يمكن أن تشير السلاسل إلى الأعمدة إذا كان الإدخال عبارة عن إطار بيانات، ويتم دائمًا تفسير الأعداد الصحيحة على أنها أعمدة موضعية.

يمكننا الاحتفاظ بأعمدة التصنيف المتبقية عن طريق تعيين ``remainder='passthrough'``.
يتم إلحاق القيم بنهاية التحويل::

  >>> column_trans = ColumnTransformer(
  ...     [('city_category', OneHotEncoder(dtype='int'),['city']),
  ...      ('title_bow', CountVectorizer(), 'title')],
  ...     remainder='passthrough')

  >>> column_trans.fit_transform(X)
  array([[1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 5, 4],
         [1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 0, 3, 5],
         [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 4, 4],
         [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 5, 3]]...)

يمكن تعيين معلمة ``remainder`` إلى مقدر لتحويل أعمدة التصنيف المتبقية.
يتم إلحاق القيم المحولة بنهاية التحويل::

  >>> from sklearn.preprocessing import MinMaxScaler
  >>> column_trans = ColumnTransformer(
  ...     [('city_category', OneHotEncoder(), ['city']),
  ...      ('title_bow', CountVectorizer(), 'title')],
  ...     remainder=MinMaxScaler())

  >>> column_trans.fit_transform(X)[:, -2:]
  array([[1. , 0.5],
         [0. , 1. ],
         [0.5, 0.5],
         [1. , 0. ]])

.. _make_column_transformer:

تتوفر دالة :func:`~sklearn.compose.make_column_transformer` لإنشاء كائن :class:`~sklearn.compose.ColumnTransformer` بسهولة أكبر.
على وجه التحديد، سيتم إعطاء الأسماء تلقائيًا.
سيكون المكافئ للمثال أعلاه هو::

  >>> from sklearn.compose import make_column_transformer
  >>> column_trans = make_column_transformer(
  ...     (OneHotEncoder(), ['city']),
  ...     (CountVectorizer(), 'title'),
  ...     remainder=MinMaxScaler())
  >>> column_trans
  ColumnTransformer(remainder=MinMaxScaler(),
                    transformers=[('onehotencoder', OneHotEncoder(), ['city']),
                                  ('countvectorizer', CountVectorizer(),
                                   'title')])

إذا تم ملاءمة :class:`~sklearn.compose.ColumnTransformer` بإطار بيانات وكان إطار البيانات يحتوي فقط على أسماء أعمدة سلسلة، فسيتم استخدام أسماء الأعمدة لتحديد الأعمدة عند تحويل إطار البيانات::


  >>> ct = ColumnTransformer(
  ...          [("scale", StandardScaler(), ["expert_rating"])]).fit(X)
  >>> X_new = pd.DataFrame({"expert_rating": [5, 6, 1],
  ...                       "ignored_new_col": [1.2, 0.3, -0.1]})
  >>> ct.transform(X_new)
  array([[ 0.9...],
         [ 2.1...],
         [-3.9...]])

.. _visualizing_composite_estimators:

تصور المقدرات المركبة
================================

يتم عرض المقدرات بتمثيل HTML عند عرضها في دفتر ملاحظات jupyter.
هذا مفيد لتشخيص أو تصور خط أنابيب مع العديد من المقدرات.
يتم تنشيط هذا التصور افتراضيًا::

  >>> column_trans  # doctest: +SKIP

يمكن إلغاء تنشيطه عن طريق تعيين خيار `display` في :func:`~sklearn.set_config` إلى "text" ::

  >>> from sklearn import set_config
  >>> set_config(display='text')  # doctest: +SKIP
  >>> # يعرض تمثيل النص في سياق jupyter
  >>> column_trans  # doctest: +SKIP

يمكن رؤية مثال على إخراج HTML في قسم **HTML representation of Pipeline** من :ref:`sphx_glr_auto_examples_compose/plot_column_transformer_mixed_types.py`.
كبديل، يمكن كتابة HTML إلى ملف باستخدام :func:`~sklearn.utils.estimator_html_repr`::

   >>> from sklearn.utils import estimator_html_repr
   >>> with open('my_estimator.html', 'w') as f:  # doctest: +SKIP
   ...     f.write(estimator_html_repr(clf))

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_compose/plot_column_transformer.py`
* :ref:`sphx_glr_auto_examples_compose/plot_column_transformer_mixed_types.py`


