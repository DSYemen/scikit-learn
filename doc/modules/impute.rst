
.. _impute:

============================
تعويض القيم المفقودة
============================

.. currentmodule:: sklearn.impute

لأسباب مُختلفة، تحتوي العديد من مجموعات البيانات في العالم الحقيقي على قيم مفقودة، غالبًا ما
تتم ترميزها على أنها فراغات أو NaNs أو عناصر نائبة أخرى. ومع ذلك، فإن مجموعات البيانات هذه
غير مُتوافقة مع مُقدِّرات scikit-learn التي تفترض أن جميع القيم في
مصفوفة هي قيم رقمية، وأن جميعها لها معنى. تتمثل الإستراتيجية الأساسية
لاستخدام مجموعات البيانات غير المكتملة في تجاهل الصفوف و / أو الأعمدة الكاملة التي تحتوي على
قيم مفقودة. ومع ذلك، يأتي هذا على حساب فقدان البيانات التي قد تكون
قيّمة (على الرغم من عدم اكتمالها). تتمثل الإستراتيجية الأفضل في تعويض القيم
المفقودة، أي استنتاجها من الجزء المعروف من البيانات. انظر
إدخال المُصطلحات في :term:`imputation`.


التعويض أحادي المتغير مقابل التعويض متعدد المتغيرات
======================================

أحد أنواع خوارزميات التعويض هو أحادي المتغير، والذي يعوض القيم في
بُعد الميزة i باستخدام القيم غير المفقودة فقط في بُعد الميزة
هذا (على سبيل المثال :class:`SimpleImputer`). على النقيض من ذلك، تستخدم خوارزميات التعويض
متعددة المتغيرات مجموعة أبعاد الميزات المتاحة بالكامل لتقدير
القيم المفقودة (على سبيل المثال :class:`IterativeImputer`).


.. _single_imputer:

تعويض الميزات أحادية المتغير
=============================

تُوفر فئة :class:`SimpleImputer` استراتيجيات أساسية لتعويض
القيم المفقودة. يمكن تعويض القيم المفقودة بقيمة ثابتة مُقدمة، أو باستخدام
إحصائيات (المتوسط أو الوسيط أو الأكثر تكرارًا) لكل عمود توجد فيه
القيم المفقودة. تسمح هذه الفئة أيضًا بترميزات قيم مفقودة
مُختلفة.

يوضح المقتطف التالي كيفية استبدال القيم المفقودة،
المُرمَّزة كـ ``np.nan``، باستخدام متوسط قيم الأعمدة (المحور 0)
التي تحتوي على القيم المفقودة::

    >>> import numpy as np
    >>> from sklearn.impute import SimpleImputer
    >>> imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    >>> imp.fit([[1, 2], [np.nan, 3], [7, 6]])
    SimpleImputer()
    >>> X = [[np.nan, 2], [6, np.nan], [7, 6]]
    >>> print(imp.transform(X))
    [[4.          2.        ]
     [6.          3.666...]
     [7.          6.        ]]

تدعم فئة :class:`SimpleImputer` أيضًا المصفوفات المتفرقة::

    >>> import scipy.sparse as sp
    >>> X = sp.csc_matrix([[1, 2], [0, -1], [8, 4]])
    >>> imp = SimpleImputer(missing_values=-1, strategy='mean')
    >>> imp.fit(X)
    SimpleImputer(missing_values=-1)
    >>> X_test = sp.csc_matrix([[-1, 2], [6, -1], [7, 6]])
    >>> print(imp.transform(X_test).toarray())
    [[3. 2.]
     [6. 3.]
     [7. 6.]]

لاحظ أن هذا التنسيق ليس مُخصصًا للاستخدام لتخزين القيم المفقودة
ضمنيًا في المصفوفة لأنه سيجعلها كثيفة في وقت التحويل. يجب استخدام
القيم المفقودة المُرمَّزة بـ 0 مع إدخال كثيف.

تدعم فئة :class:`SimpleImputer` أيضًا البيانات الفئوية المُمثلة كـ
قيم سلسلة أو فئات pandas عند استخدام استراتيجية ``'most_frequent'`` أو
``'constant'``::

    >>> import pandas as pd
    >>> df = pd.DataFrame([["a", "x"],
    ...                    [np.nan, "y"],
    ...                    ["a", np.nan],
    ...                    ["b", "y"]], dtype="category")
    ...
    >>> imp = SimpleImputer(strategy="most_frequent")
    >>> print(imp.fit_transform(df))
    [['a' 'x']
     ['a' 'y']
     ['a' 'y']
     ['b' 'y']]

لمثال آخر عن الاستخدام، انظر :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.

.. _iterative_imputer:


تعويض الميزات متعددة المتغيرات
===============================

النهج الأكثر تعقيدًا هو استخدام فئة :class:`IterativeImputer`،
التي تُنمذج كل ميزة ذات قيم مفقودة كدالة لميزات أخرى،
وتستخدم هذا التقدير للتعويض. يفعل ذلك بطريقة دائرية متكررة: في كل خطوة،
يتم تعيين عمود ميزة كناتج ``y`` و
يتم التعامل مع أعمدة الميزات الأخرى كمدخلات ``X``. يتم ملاءمة مُنحدِر على ``(X,
y)`` لـ ``y`` المعروف. ثم، يتم استخدام المُنحدِر للتنبؤ بالقيم المفقودة
لـ ``y``. يتم ذلك لكل ميزة بطريقة متكررة، ثم يتم
تكراره لـ ``max_iter`` جولات تعويض. يتم إرجاع نتائج جولة
التعويض الأخيرة.

.. note::

   لا يزال هذا المُقدِّر **تجريبيًا** في الوقت الحالي: قد تتغير المعلمات الافتراضية أو
   تفاصيل السلوك دون أي دورة إهمال. سيؤدي حل
   المشاكل التالية إلى استقرار :class:`IterativeImputer`:
   معايير التقارب (:issue:`14338`)، المُقدِّرات الافتراضية (:issue:`13286`)،
   واستخدام الحالة العشوائية (:issue:`15611`). لاستخدامه، تحتاج إلى
   استيراد ``enable_iterative_imputer`` صراحةً.

::

    >>> import numpy as np
    >>> from sklearn.experimental import enable_iterative_imputer
    >>> from sklearn.impute import IterativeImputer
    >>> imp = IterativeImputer(max_iter=10, random_state=0)
    >>> imp.fit([[1, 2], [3, 6], [4, 8], [np.nan, 3], [7, np.nan]])
    IterativeImputer(random_state=0)
    >>> X_test = [[np.nan, 2], [6, np.nan], [np.nan, 6]]
    >>> # يتعلم النموذج أن الميزة الثانية هي ضعف الميزة الأولى
    >>> print(np.round(imp.transform(X_test)))
    [[ 1.  2.]
     [ 6. 12.]
     [ 3.  6.]]

يمكن استخدام كل من :class:`SimpleImputer` و :class:`IterativeImputer` في
خط أنابيب كطريقة لبناء مُقدِّر مُركب يدعم التعويض.
انظر :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.

مرونة IterativeImputer
-------------------------------

هناك العديد من حزم التعويض الراسخة في نظام R البيئي لعلوم
البيانات: Amelia و mi و mice و missForest، إلخ. missForest شائع، ويتبين
أنه مثيل مُعين لخوارزميات تعويض متسلسلة مُختلفة
والتي يمكن تطبيقها جميعًا باستخدام :class:`IterativeImputer` عن طريق تمرير
مُنحدرات مُختلفة لاستخدامها في التنبؤ بقيم الميزات المفقودة. في
حالة missForest، هذا المُنحدِر هو غابة عشوائية.
انظر :ref:`sphx_glr_auto_examples_impute_plot_iterative_imputer_variants_comparison.py`.


.. _multiple_imputation:

التعويض المتعدد مقابل التعويض الفردي
------------------------------

في مجتمع الإحصاء، من الشائع إجراء تعويضات متعددة،
توليد، على سبيل المثال، ``m`` تعويضات مُنفصلة لمصفوفة
ميزة واحدة. يتم بعد ذلك وضع كل من هذه التعويضات ``m`` من خلال
خط أنابيب التحليل اللاحق (على سبيل المثال هندسة الميزات، التجميع،
الانحدار، التصنيف). تسمح نتائج التحليل النهائية ``m`` (على سبيل المثال
أخطاء التحقق من الصحة المُخصصة للاختبار) لعالم البيانات بالحصول على فهم
لكيفية اختلاف نتائج التحليل كنتيجة لعدم اليقين المتأصل الناجم عن
القيم المفقودة. تُسمى الممارسة المذكورة أعلاه التعويض المتعدد.

استوحى تطبيقنا لـ :class:`IterativeImputer` من حزمة R MICE
(التعويض المتعدد بواسطة المعادلات المتسلسلة) [1]_، ولكنه يختلف عنها
من خلال إرجاع تعويض فردي بدلاً من تعويضات متعددة. ومع ذلك،
يمكن أيضًا استخدام :class:`IterativeImputer` لتعويضات متعددة عن طريق تطبيقه
بشكل متكرر على نفس مجموعة البيانات ببذور عشوائية مختلفة عندما
``sample_posterior=True``. انظر [2]_، الفصل 4 لمزيد من المناقشة حول
التعويضات المتعددة مقابل الفردية.

لا تزال مشكلة مفتوحة حول مدى فائدة التعويض الفردي مقابل المتعدد
في سياق التنبؤ والتصنيف عندما لا يكون المستخدم
مهتمًا بقياس عدم اليقين بسبب القيم المفقودة.

لاحظ أن استدعاء أسلوب ``transform`` لـ :class:`IterativeImputer`
غير مسموح به لتغيير عدد العينات. لذلك لا يمكن تحقيق
تعويضات متعددة باستدعاء واحد لـ ``transform``.

المراجع
----------

.. [1] `Stef van Buuren, Karin Groothuis-Oudshoorn (2011). "mice: التعويض
   المتعدد بواسطة المعادلات المتسلسلة في R". Journal of Statistical Software 45:
   1-67. <https://www.jstatsoft.org/article/view/v045i03>`_

.. [2] Roderick J A Little and Donald B Rubin (1986). "التحليل الإحصائي
   مع البيانات المفقودة". John Wiley & Sons, Inc., New York, NY, USA.

.. _knnimpute:

تعويض أقرب الجيران
============================

تُوفر فئة :class:`KNNImputer` تعويضًا لملء القيم المفقودة
باستخدام نهج أقرب جيران k. افتراضيًا، مقياس مسافة إقليدية
يدعم القيم المفقودة،
:func:`~sklearn.metrics.pairwise.nan_euclidean_distances`، يُستخدم للعثور على
أقرب الجيران. يتم تعويض كل ميزة مفقودة باستخدام قيم من
``n_neighbors`` أقرب جيران لها قيمة للميزة. يتم حساب متوسط
ميزة الجيران بشكل مُوحد أو موزون بالمسافة إلى كل
جار. إذا كانت العينة تحتوي على أكثر من ميزة مفقودة، فقد يكون الجيران
لهذه العينة مُختلفين اعتمادًا على الميزة المُعينة التي يتم تعويضها.
عندما يكون عدد الجيران المتاحين أقل من `n_neighbors` ولا توجد
مسافات مُحددة لمجموعة التدريب، يتم استخدام متوسط مجموعة التدريب لتلك
الميزة أثناء التعويض. إذا كان هناك جار واحد على الأقل بـ
مسافة مُحددة، فسيتم استخدام المتوسط الموزون أو غير الموزون للجيران المتبقيين
أثناء التعويض. إذا كانت الميزة مفقودة دائمًا في التدريب، فسيتم إزالتها
أثناء `transform`. لمزيد من المعلومات حول المنهجية، انظر
المرجع [OL2001]_.

يوضح المقتطف التالي كيفية استبدال القيم المفقودة،
المُرمَّزة كـ ``np.nan``، باستخدام متوسط قيمة الميزة لأقرب جارين
من العينات ذات القيم المفقودة::

    >>> import numpy as np
    >>> from sklearn.impute import KNNImputer
    >>> nan = np.nan
    >>> X = [[1, 2, nan], [3, 4, 3], [nan, 6, 5], [8, 8, 7]]
    >>> imputer = KNNImputer(n_neighbors=2, weights="uniform")
    >>> imputer.fit_transform(X)
    array([[1. , 2. , 4. ],
           [3. , 4. , 3. ],
           [5.5, 6. , 5. ],
           [8. , 8. , 7. ]])

لمثال آخر عن الاستخدام، انظر :ref:`sphx_glr_auto_examples_impute_plot_missing_values.py`.

.. rubric:: المراجع

.. [OL2001] `Olga Troyanskaya, Michael Cantor, Gavin Sherlock, Pat Brown,
    Trevor Hastie, Robert Tibshirani, David Botstein and Russ B. Altman,
    أساليب تقدير القيمة المفقودة لرقائق الحمض النووي الدقيقة، BIOINFORMATICS
    المجلد 17 رقم 6، 2001 الصفحات 520-525.
    <https://academic.oup.com/bioinformatics/article/17/6/520/272365>`_

الحفاظ على عدد الميزات ثابتًا
=======================================

افتراضيًا، ستُسقط مُعوضات scikit-learn الميزات الفارغة تمامًا، أي
الأعمدة التي تحتوي على قيم مفقودة فقط. على سبيل المثال::

  >>> imputer = SimpleImputer()
  >>> X = np.array([[np.nan, 1], [np.nan, 2], [np.nan, 3]])
  >>> imputer.fit_transform(X)
  array([[1.],
         [2.],
         [3.]])

تم إسقاط الميزة الأولى في `X` التي تحتوي فقط على `np.nan` بعد
التعويض. بينما لن تُساعد هذه الميزة في إعداد التنبؤ، فإن إسقاط
الأعمدة سيُغير شكل `X` الذي قد يكون مُشكلًا عند استخدام
المُعوضات في خط أنابيب تعلم آلي أكثر تعقيدًا. تُوفر المعلمة
`keep_empty_features` خيار الاحتفاظ بالميزات الفارغة عن طريق التعويض
بقيم ثابتة. في معظم الحالات، تكون هذه القيمة الثابتة صفرًا::

  >>> imputer.set_params(keep_empty_features=True)
  SimpleImputer(keep_empty_features=True)
  >>> imputer.fit_transform(X)
  array([[0., 1.],
         [0., 2.],
         [0., 3.]])

.. _missing_indicator:

وضع علامة على القيم المُعوضة
======================

مُحوِّل :class:`MissingIndicator` مفيد لتحويل مجموعة بيانات إلى
مصفوفة ثنائية مُقابلة تُشير إلى وجود قيم مفقودة في
مجموعة البيانات. هذا التحويل مفيد بالاقتران مع التعويض. عند
استخدام التعويض، يمكن أن يكون الاحتفاظ بالمعلومات حول القيم التي كانت
مفقودة مفيدًا. لاحظ أن كلاً من :class:`SimpleImputer` و
:class:`IterativeImputer` لديهما معلمة منطقية ``add_indicator``
(``False`` افتراضيًا) والتي عند تعيينها إلى ``True`` تُوفر طريقة مُريحة لـ
تكديس ناتج مُحوِّل :class:`MissingIndicator` مع
ناتج المُعوض.

عادةً ما يُستخدم ``NaN`` كعنصر نائب للقيم المفقودة. ومع ذلك، فإنه
يفرض أن يكون نوع البيانات عائمًا. تسمح المعلمة ``missing_values`` بـ
تحديد عنصر نائب آخر مثل عدد صحيح. في المثال التالي، سنستخدم
``-1`` كقيم مفقودة::

  >>> from sklearn.impute import MissingIndicator
  >>> X = np.array([[-1, -1, 1, 3],
  ...               [4, -1, 0, -1],
  ...               [8, -1, 1, 0]])
  >>> indicator = MissingIndicator(missing_values=-1)
  >>> mask_missing_values_only = indicator.fit_transform(X)
  >>> mask_missing_values_only
  array([[ True,  True, False],
         [False,  True,  True],
         [False,  True, False]])

تُستخدم معلمة ``features`` لاختيار الميزات التي يتم إنشاء القناع
من أجلها. افتراضيًا، هي ``'missing-only'`` التي تُعيد قناع المُعوض
للميزات التي تحتوي على قيم مفقودة في وقت ``fit``::

  >>> indicator.features_
  array([0, 1, 3])

يمكن تعيين معلمة ``features`` إلى ``'all'`` لإعادة جميع الميزات
سواء كانت تحتوي على قيم مفقودة أم لا::

  >>> indicator = MissingIndicator(missing_values=-1, features="all")
  >>> mask_all = indicator.fit_transform(X)
  >>> mask_all
  array([[ True,  True, False, False],
         [False,  True, False,  True],
         [False,  True, False, False]])
  >>> indicator.features_
  array([0, 1, 2, 3])

عند استخدام :class:`MissingIndicator` في
:class:`~sklearn.pipeline.Pipeline`، تأكد من استخدام
:class:`~sklearn.pipeline.FeatureUnion` أو
:class:`~sklearn.compose.ColumnTransformer` لإضافة ميزات المؤشر إلى
الميزات العادية. أولاً، نحصل على مجموعة بيانات `iris`، ونضيف بعض القيم
المفقودة إليها.

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.impute import SimpleImputer, MissingIndicator
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn.pipeline import FeatureUnion, make_pipeline
  >>> from sklearn.tree import DecisionTreeClassifier
  >>> X, y = load_iris(return_X_y=True)
  >>> mask = np.random.randint(0, 2, size=X.shape).astype(bool)
  >>> X[mask] = np.nan
  >>> X_train, X_test, y_train, _ = train_test_split(X, y, test_size=100,
  ...                                                random_state=0)

الآن ننشئ :class:`~sklearn.pipeline.FeatureUnion`. سيتم تعويض جميع الميزات
باستخدام :class:`SimpleImputer`، من أجل تمكين المُصنِّفات من العمل
مع هذه البيانات. بالإضافة إلى ذلك، فإنه يُضيف متغيرات المؤشر من
:class:`MissingIndicator`.

  >>> transformer = FeatureUnion(
  ...     transformer_list=[
  ...         ('features', SimpleImputer(strategy='mean')),
  ...         ('indicators', MissingIndicator())])
  >>> transformer = transformer.fit(X_train, y_train)
  >>> results = transformer.transform(X_test)
  >>> results.shape
  (100, 8)

بالطبع، لا يمكننا استخدام المُحوِّل لإجراء أي تنبؤات. يجب علينا
تضمين هذا في :class:`~sklearn.pipeline.Pipeline` مع مُصنف (على سبيل المثال،
:class:`~sklearn.tree.DecisionTreeClassifier`) لتكون قادرًا على إجراء تنبؤات.

  >>> clf = make_pipeline(transformer, DecisionTreeClassifier())
  >>> clf = clf.fit(X_train, y_train)
  >>> results = clf.predict(X_test)
  >>> results.shape
  (100,)

المُقدِّرات التي تتعامل مع قيم NaN
=================================

تم تصميم بعض المُقدِّرات للتعامل مع قيم NaN بدون مُعالجة مُسبقة.
فيما يلي قائمة بهذه المُقدِّرات، مُصنفة حسب النوع
(التجميع، المُنحدِر، المُصنف، التحويل):

.. allow_nan_estimators::



