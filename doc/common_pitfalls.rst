
.. _common_pitfalls:

=========================================
المخاطر الشائعة والممارسات الموصى بها
=========================================

الغرض من هذا الفصل هو توضيح بعض المخاطر الشائعة والأنماط 
غير الصحيحة التي تحدث عند استخدام Scikit-learn. يقدم
أمثلة على ما **يجب عدم فعله**، جنبًا إلى جنب مع مثال صحيح 
مقابل.

معالجة مسبقة غير متسقة
==========================

يوفر Scikit-learn مكتبة من :ref:`تحويلات البيانات <data-transforms>`، والتي
قد تنظف (انظر :ref:`preprocessing`)، أو تقلل
(انظر :ref:`data_reduction`)، أو توسع (انظر :ref:`kernel_approximation`)
أو تولد (انظر :ref:`feature_extraction`) تمثيلات الميزات.
إذا تم استخدام تحويلات البيانات هذه عند تدريب نموذج، فيجب أيضًا
استخدامها على مجموعات البيانات اللاحقة، سواء كانت بيانات اختبار أو
بيانات في نظام إنتاج. خلاف ذلك، ستتغير مساحة الميزات،
ولن يكون النموذج قادرًا على الأداء بشكل فعال.

للحصول على المثال التالي، لنقم بإنشاء مجموعة بيانات اصطناعية مع
ميزة واحدة::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split

    >>> random_state = 42
    >>> X, y = make_regression(random_state=random_state, n_features=1, noise=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, random_state=random_state)

**خطأ**

يتم قياس مجموعة بيانات التدريب، ولكن ليس مجموعة بيانات الاختبار، لذلك نموذج
الأداء على مجموعة بيانات الاختبار أسوأ مما كان متوقعًا::

    >>> from sklearn.metrics import mean_squared_error
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.preprocessing import StandardScaler

    >>> scaler = StandardScaler()
    >>> X_train_transformed = scaler.fit_transform(X_train)
    >>> model = LinearRegression().fit(X_train_transformed, y_train)
    >>> mean_squared_error(y_test, model.predict(X_test))
    62.80...

**صحيح**

بدلاً من تمرير `X_test` غير المحولة إلى `predict`، يجب علينا
تحويل بيانات الاختبار، بنفس الطريقة التي حولنا بها بيانات التدريب::

    >>> X_test_transformed = scaler.transform(X_test)
    >>> mean_squared_error(y_test, model.predict(X_test_transformed))
    0.90...

بدلاً من ذلك، نوصي باستخدام :class:`Pipeline
<sklearn.pipeline.Pipeline>`، مما يسهل ربط التحويلات
مع المقدرات، ويقلل من احتمالية نسيان التحويل::

    >>> from sklearn.pipeline import make_pipeline

    >>> model = make_pipeline(StandardScaler(), LinearRegression())
    >>> model.fit(X_train, y_train)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('linearregression', LinearRegression())])
    >>> mean_squared_error(y_test, model.predict(X_test))
    0.90...

تساعد خطوط الأنابيب أيضًا في تجنب خطر شائع آخر: تسريب بيانات الاختبار
إلى بيانات التدريب.

.. _data_leakage:

تسريب البيانات
============

يحدث تسريب البيانات عندما يتم استخدام معلومات لن تكون متاحة في وقت التنبؤ
عند بناء النموذج. ينتج عن هذا تقديرات أداء متفائلة للغاية، على سبيل المثال من :ref:`التحقق المتبادل
<cross_validation>`، وبالتالي أداء أسوأ عندما يتم استخدام النموذج
على بيانات جديدة فعليًا، على سبيل المثال أثناء الإنتاج.

سبب شائع هو عدم إبقاء مجموعات بيانات الاختبار والتدريب منفصلة.
لا ينبغي أبدًا استخدام بيانات الاختبار لاتخاذ قرارات بشأن النموذج.
**القاعدة العامة هي عدم الاتصال أبدًا** `fit` **على بيانات الاختبار**. بينما هذا
قد يبدو واضحًا، من السهل تفويته في بعض الحالات، على سبيل المثال عند
تطبيق خطوات معالجة مسبقة معينة.

على الرغم من أنه يجب أن تتلقى كل من مجموعات بيانات التدريب والاختبار نفس
تحويل المعالجة المسبقة (كما هو موضح في القسم السابق)، من المهم أن تكون هذه التحويلات
تم تعلمها فقط من بيانات التدريب.
على سبيل المثال، إذا كانت لديك خطوة
تطبيع حيث تقسم على القيمة المتوسطة، فيجب أن يكون المتوسط
هو متوسط مجموعة التدريب الفرعية، **وليس** متوسط جميع البيانات. إذا كان
مجموعة الاختبار الفرعية مدرجة في حساب المتوسط، معلومات من مجموعة الاختبار الفرعية
تؤثر على النموذج.

كيفية تجنب تسرب البيانات
-------------------------

فيما يلي بعض النصائح حول تجنب تسرب البيانات:

* قم دائمًا بتقسيم البيانات إلى مجموعات تدريب واختبار فرعية أولاً، خاصةً
  قبل أي خطوات معالجة مسبقة.
* لا تقم أبدًا بتضمين بيانات الاختبار عند استخدام `fit` و `fit_transform`
  الطرق. باستخدام جميع البيانات، على سبيل المثال، `fit(X)`، يمكن أن يؤدي إلى نتائج متفائلة للغاية.

  على العكس من ذلك، يجب استخدام طريقة `transform` على كل من مجموعات التدريب والاختبار الفرعية مثل
  يجب تطبيق نفس المعالجة المسبقة على جميع البيانات.
  يمكن تحقيق ذلك باستخدام `fit_transform` على مجموعة التدريب الفرعية و
  `transform` على مجموعة الاختبار الفرعية.
* Scikit-learn :ref:`pipeline <pipeline>` هي طريقة رائعة لمنع تسرب البيانات كما هي
  يضمن تنفيذ الطريقة المناسبة على
  مجموعة بيانات فرعية صحيحة. خط الأنابيب مثالي للاستخدام في التحقق المتبادل
  وظائف ضبط المعلمات الفائقة.

يتم وصف مثال على تسرب البيانات أثناء المعالجة المسبقة أدناه.

تسرب البيانات أثناء المعالجة المسبقة
----------------------------------

.. note::
    نختار هنا توضيح تسرب البيانات بخطوة اختيار الميزات.
    ومع ذلك، فإن خطر التسرب هذا وثيق الصلة بجميع التحويلات تقريبًا
    في Scikit-learn، بما في ذلك (على سبيل المثال لا الحصر)
    :class:`~sklearn.preprocessing.StandardScaler`،
    :class:`~sklearn.impute.SimpleImputer`، و
    :class:`~sklearn.decomposition.PCA`.

يتوفر عدد من وظائف :ref:`feature_selection` في Scikit-learn.
يمكنهم المساعدة في إزالة الميزات غير ذات الصلة والزائدة عن الحاجة والضوضاء وكذلك
تحسين وقت إنشاء النموذج وأدائه. كما هو الحال مع أي نوع آخر من
المعالجة المسبقة، يجب أن يستخدم اختيار الميزات **فقط** بيانات التدريب.
سيؤدي تضمين بيانات الاختبار في اختيار الميزات إلى تحيز نموذجك بشكل متفائل.

للتوضيح، سننشئ مشكلة تصنيف ثنائية هذه باستخدام
10000 ميزة تم إنشاؤها عشوائيًا::

    >>> import numpy as np
    >>> n_samples, n_features, n_classes = 200, 10000, 2
    >>> rng = np.random.RandomState(42)
    >>> X = rng.standard_normal((n_samples, n_features))
    >>> y = rng.choice(n_classes, n_samples)

**خطأ**

يؤدي استخدام جميع البيانات لإجراء اختيار الميزات إلى نتيجة دقة
أعلى بكثير من الصدفة، على الرغم من أن أهدافنا عشوائية تمامًا.
هذه العشوائية تعني أن `X` و `y` لدينا مستقلان وبالتالي نتوقع
أن تكون الدقة حوالي 0.5. ومع ذلك، نظرًا لأن خطوة اختيار الميزات
"ترى" بيانات الاختبار، فإن النموذج يتمتع بميزة غير عادلة. في غير صحيح
في المثال أدناه، نستخدم أولاً جميع البيانات لاختيار الميزات ثم نقسم
البيانات إلى مجموعات تدريب واختبار فرعية لملاءمة النموذج. النتيجة هي
نتيجة دقة أعلى بكثير من المتوقع::

    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.metrics import accuracy_score

    >>> # معالجة مسبقة غير صحيحة: يتم تحويل البيانات بالكامل
    >>> X_selected = SelectKBest(k=25).fit_transform(X, y)

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X_selected, y, random_state=42)
    >>> gbc = GradientBoostingClassifier(random_state=1)
    >>> gbc.fit(X_train, y_train)
    GradientBoostingClassifier(random_state=1)

    >>> y_pred = gbc.predict(X_test)
    >>> accuracy_score(y_test, y_pred)
    0.76

**صحيح**

لمنع تسرب البيانات، من الممارسات الجيدة تقسيم بياناتك إلى مجموعات تدريب
ومجموعات اختبار فرعية **أولاً**. يمكن بعد ذلك تشكيل اختيار الميزات باستخدام فقط
مجموعة بيانات التدريب. لاحظ أنه كلما استخدمنا `fit` أو `fit_transform`، نحن
استخدم مجموعة بيانات التدريب فقط. النتيجة الآن هي ما نتوقعه من أجل
البيانات، قريبة من الصدفة::

    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42)
    >>> select = SelectKBest(k=25)
    >>> X_train_selected = select.fit_transform(X_train, y_train)

    >>> gbc = GradientBoostingClassifier(random_state=1)
    >>> gbc.fit(X_train_selected, y_train)
    GradientBoostingClassifier(random_state=1)

    >>> X_test_selected = select.transform(X_test)
    >>> y_pred = gbc.predict(X_test_selected)
    >>> accuracy_score(y_test, y_pred)
    0.46

هنا مرة أخرى، نوصي باستخدام :class:`~sklearn.pipeline.Pipeline` لتسلسل
معًا مقدرات اختيار الميزات والنموذج. يضمن خط الأنابيب
أنه يتم استخدام بيانات التدريب فقط عند أداء `fit` وبيانات الاختبار
يتم استخدامه فقط لحساب درجة الدقة::

    >>> from sklearn.pipeline import make_pipeline
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, random_state=42)
    >>> pipeline = make_pipeline(SelectKBest(k=25),
    ...                          GradientBoostingClassifier(random_state=1))
    >>> pipeline.fit(X_train, y_train)
    Pipeline(steps=[('selectkbest', SelectKBest(k=25)),
                    ('gradientboostingclassifier',
                    GradientBoostingClassifier(random_state=1))])

    >>> y_pred = pipeline.predict(X_test)
    >>> accuracy_score(y_test, y_pred)
    0.46

يمكن أيضًا تغذية خط الأنابيب في التحقق المتبادل
وظيفة مثل :func:`~sklearn.model_selection.cross_val_score`.
مرة أخرى، يضمن خط الأنابيب أن مجموعة البيانات الفرعية الصحيحة ومقدر
يتم استخدام الطريقة أثناء التركيب والتنبؤ::

    >>> from sklearn.model_selection import cross_val_score
    >>> scores = cross_val_score(pipeline, X, y)
    >>> print(f"متوسط الدقة: {scores.mean():.2f}+/-{scores.std():.2f}")
    متوسط الدقة: 0.46+/-0.07


.. _randomness:

التحكم في العشوائية
======================

بعض كائنات Scikit-learn عشوائية بطبيعتها. عادة ما تكون هذه مقدرات
(على سبيل المثال :class:`~sklearn.ensemble.RandomForestClassifier`) ومقسمات التحقق المتبادل
(على سبيل المثال :class:`~sklearn.model_selection.KFold`). عشوائية
يتم التحكم في هذه الكائنات عبر معلمة `random_state` الخاصة بها، كما هو موضح
في :term:`المسرد <random_state>`. يوسع هذا القسم إدخال المسرد،
ويصف الممارسات الجيدة والمخاطر الشائعة فيما يتعلق بهذا
معلمة دقيقة.

.. note:: ملخص التوصية

    للحصول على متانة مثالية لنتائج التحقق المتبادل (CV)، مرر
    `RandomState` مثيلات عند إنشاء المقدرات، أو اترك `random_state`
    إلى `None`. عادةً ما يكون تمرير الأعداد الصحيحة إلى مقسمات CV هو الخيار الأكثر أمانًا
    وهو الأفضل؛ تمرير `RandomState` مثيلات إلى الموصلات قد
    في بعض الأحيان تكون مفيدة لتحقيق حالات استخدام محددة للغاية.
    بالنسبة لكل من المقدرات والمقسمات، تمرير عدد صحيح مقابل تمرير ملف
    مثيل (أو `None`) يؤدي إلى اختلافات دقيقة ولكنها مهمة،
    خاصة بالنسبة لإجراءات CV. من المهم فهم هذه الاختلافات
    عند الإبلاغ عن النتائج.

    للحصول على نتائج قابلة للتكرار عبر عمليات التنفيذ، قم بإزالة أي استخدام لـ
    `random_state=None`.

استخدام `None` أو `RandomState` مثيلات، والمكالمات المتكررة إلى `fit` و `split`
--------------------------------------------------------------------------------

تحدد معلمة `random_state` ما إذا كانت المكالمات المتعددة لـ :term:`fit`
(للمقدرات) أو لـ :term:`split` (للمقسمات CV) ستنتج نفس الشيء
النتائج، وفقًا لهذه القواعد:

- إذا تم تمرير عدد صحيح، فإن الاتصال بـ `fit` أو `split` عدة مرات دائمًا
  ينتج نفس النتائج.
- إذا تم تمرير `None` أو `RandomState` مثيل: `fit` و `split` سوف
  ينتج نتائج مختلفة في كل مرة يتم استدعاؤها، وتوالي
  تستكشف المكالمات جميع مصادر الانتروبيا. `None` هي القيمة الافتراضية للجميع
  معلمات `random_state`.

نوضح هنا هذه القواعد لكل من المقدرات ومقسمات CV.

.. note::
    نظرًا لأن تمرير `random_state=None` يعادل تمرير العام
    `RandomState` مثيل من `numpy`
    (`random_state=np.random.mtrand._rand`)، فلن نذكر صراحة
    `None` هنا. كل ما ينطبق على المثيلات ينطبق أيضًا على الاستخدام
    `None`.

مقدرات
..........

تمرير المثيلات يعني أن الاتصال بـ `fit` عدة مرات لن ينتج عنه
نفس النتائج، حتى لو كان المقدر مناسبًا على نفس البيانات ومع نفس الشيء
المعلمات الفائقة::

    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np

    >>> rng = np.random.RandomState(0)
    >>> X, y = make_classification(n_features=5, random_state=rng)
    >>> sgd = SGDClassifier(random_state=rng)

    >>> sgd.fit(X, y).coef_
    array([[ 8.85418642,  4.79084103, -3.13077794,  8.11915045, -0.56479934]])

    >>> sgd.fit(X, y).coef_
    array([[ 6.70814003,  5.25291366, -7.55212743,  5.18197458,  1.37845099]])

يمكننا أن نرى من المقتطف أعلاه أن الاتصال بـ `sgd.fit` بشكل متكرر
أنتجت نماذج مختلفة، حتى لو كانت البيانات هي نفسها. هذا لأن
يتم استهلاك مولد الأرقام العشوائية (RNG) للمقدر (أي تحوره)
عندما يتم استدعاء `fit`، وسيتم استخدام RNG المتحور هذا في اللاحقة
يدعو إلى `fit`. بالإضافة إلى ذلك، يتم مشاركة كائن `rng` عبر جميع الكائنات
التي تستخدمه، ونتيجة لذلك، تصبح هذه الكائنات مترابطة إلى حد ما.
على سبيل المثال، اثنين من المقدرات التي تشترك في نفس الشيء
سيؤثر مثيل `RandomState` على بعضهما البعض، كما سنرى لاحقًا عند
ناقش الاستنساخ. من المهم مراعاة هذه النقطة عند التصحيح.

إذا مررنا عددًا صحيحًا إلى معلمة `random_state` لـ
:class:`~sklearn.linear_model.SGDClassifier`، لكنا قد حصلنا على
نفس النماذج، وبالتالي نفس الدرجات في كل مرة. عندما نمرر عددًا صحيحًا،
يتم استخدام نفس RNG عبر جميع المكالمات إلى `fit`. ما يحدث داخليًا هو أنه
على الرغم من استهلاك RNG عند استدعاء `fit`، إلا أنه يتم دائمًا إعادة تعيينه إلى
حالتها الأصلية في بداية `fit`.

مقسمات CV
............

مقسمات CV العشوائية لها سلوك مشابه عندما يكون `RandomState`
يتم تمرير مثيل؛ ينتج عن استدعاء `split` عدة مرات بيانات مختلفة
ينقسم::

    >>> from sklearn.model_selection import KFold
    >>> import numpy as np

    >>> X = y = np.arange(10)
    >>> rng = np.random.RandomState(0)
    >>> cv = KFold(n_splits=2, shuffle=True, random_state=rng)

    >>> for train, test in cv.split(X, y):
    ...     print(train, test)
    [0 3 5 6 7] [1 2 4 8 9]
    [1 2 4 8 9] [0 3 5 6 7]

    >>> for train, test in cv.split(X, y):
    ...     print(train, test)
    [0 4 6 7 8] [1 2 3 5 9]
    [1 2 3 5 9] [0 4 6 7 8]

يمكننا أن نرى أن الانقسامات مختلفة عن المرة الثانية التي يتم فيها `split`
تم استدعاؤه. قد يؤدي هذا إلى نتائج غير متوقعة إذا قارنت أداء
مقدمات متعددة عن طريق استدعاء `split` عدة مرات، كما سنرى في القسم التالي.

المخاطر الشائعة والدقيقة
------------------------------

في حين أن القواعد التي تحكم معلمة `random_state` تبدو بسيطة،
ومع ذلك، فإن لها بعض الآثار الدقيقة. في بعض الحالات، هذا يمكن أن
حتى تؤدي إلى استنتاجات خاطئة.

مقدرات
..........

**أنواع `random_state` المختلفة تؤدي إلى اختلاف التحقق المتبادل
الإجراءات**

اعتمادًا على نوع معلمة `random_state`، ستتصرف المقدرات
بشكل مختلف، خاصة في إجراءات التحقق المتبادل. اعتبر
مقتطف التالي::

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import cross_val_score
    >>> import numpy as np

    >>> X, y = make_classification(random_state=0)

    >>> rf_123 = RandomForestClassifier(random_state=123)
    >>> cross_val_score(rf_123, X, y)
    array([0.85, 0.95, 0.95, 0.9 , 0.9 ])

    >>> rf_inst = RandomForestClassifier(random_state=np.random.RandomState(0))
    >>> cross_val_score(rf_inst, X, y)
    array([0.9 , 0.95, 0.95, 0.9 , 0.9 ])

نرى أن درجات التحقق المتبادل لـ `rf_123` و `rf_inst` هي
مختلفة، كما هو متوقع لأننا لم نمرر نفس `random_state`
معامل. ومع ذلك، فإن الفرق بين هذه الدرجات أكثر دقة مما
يبدو، و **إجراءات التحقق المتبادل التي تم تنفيذها بواسطة**
:func:`~sklearn.model_selection.cross_val_score` **تختلف اختلافًا كبيرًا في
كل حالة**:

- نظرًا لأنه تم تمرير عدد صحيح إلى `rf_123`، فإن كل مكالمة إلى `fit` تستخدم نفس RNG:
  هذا يعني أن جميع الخصائص العشوائية لمقدر الغابة العشوائية
  ستكون هي نفسها لكل من 5 طيات من إجراء CV. في
  على وجه الخصوص، ستكون المجموعة الفرعية (المختارة عشوائيًا) من سمات المقدر
  نفس الشيء عبر جميع الطيات.
- نظرًا لأنه تم تمرير `RandomState` مثيل إلى `rf_inst`، فإن كل مكالمة إلى `fit`
  يبدأ من RNG مختلف. نتيجة لذلك، مجموعة فرعية عشوائية من الميزات
  سيكون مختلفًا لكل طيات.

بينما وجود مقدر RNG ثابت عبر الطيات ليس خطأ بطبيعته، فنحن
عادة ما تريد نتائج CV التي تكون قوية فيما يتعلق بعشوائية المقدر. كما
نتيجة لذلك، قد يكون تمرير مثيل بدلاً من عدد صحيح أمرًا مفضلًا، لأنه
سيسمح للمقدر RNG بالتنوع لكل طية.

.. note::
    هنا، :func:`~sklearn.model_selection.cross_val_score` سيستخدم ملف
    مقسم CV غير عشوائي (كما هو افتراضي)، لذلك سيتم كلا المقدرين
    يتم تقييمها على نفس الانقسامات. لا يتعلق هذا القسم بالتنوع في
    الانقسامات. أيضًا، سواء مررنا عددًا صحيحًا أو مثيلًا إلى
    :func:`~sklearn.datasets.make_classification` غير ذي صلة بغرضنا التوضيحي: ما يهم هو ما نمرره إلى
    :class:`~sklearn.ensemble.RandomForestClassifier` مقدر.

.. dropdown:: الاستنساخ

    أثر جانبي دقيق آخر لتمرير `RandomState` مثيلات هو كيف
    :func:`~sklearn.base.clone` سيعمل::

        >>> from sklearn import clone
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np

        >>> rng = np.random.RandomState(0)
        >>> a = RandomForestClassifier(random_state=rng)
        >>> b = clone(a)

    نظرًا لأنه تم تمرير `RandomState` مثيل إلى `a`، فإن `a` و `b` ليسا مستنسخين
    بالمعنى الدقيق للكلمة، بل بالأحرى مستنسخين بالمعنى الإحصائي: `a` و `b`
    ستظل نماذج مختلفة، حتى عند استدعاء `fit(X، y)` على نفس الشيء
    البيانات. علاوة على ذلك، سيؤثر `a` و `b` على بعضهما البعض لأنهما يشتركان في
    نفس RNG الداخلي: استدعاء `a.fit` سيستهلك RNG لـ `b`، واستدعاء
    `b.fit` سيستهلك RNG لـ `a`، لأنها متشابهة. هذه القطعة صحيحة لـ
    أي مقدرات تشترك في معلمة `random_state`؛ انها ليست محددة ل
    المستنسخات.

    إذا تم تمرير عدد صحيح، فسيكون `a` و `b` مستنسخين دقيقين ولن يكونوا كذلك
    تؤثر على بعضها البعض.

    .. warning::
        على الرغم من أن :func:`~sklearn.base.clone` نادرًا ما يتم استخدامه في كود المستخدم، إلا أنه كذلك
        تم استدعاؤها بشكل شامل في جميع أنحاء قاعدة كود Scikit-learn: على وجه الخصوص، معظم
        المقدرات الوصفية التي تقبل المقدرات غير المجهزة تدعو
        :func:`~sklearn.base.clone` داخليًا
        (:class:`~sklearn.model_selection.GridSearchCV`،
        :class:`~sklearn.ensemble.StackingClassifier`،
        :class:`~sklearn.calibration.CalibratedClassifierCV`، إلخ.).


مقسمات CV
............

عند تمرير `RandomState` مثيل، تنتج مقسمات CV انقسامات مختلفة
في كل مرة يتم استدعاء `split`. عند مقارنة مقدرات مختلفة، يمكن لهذا
لتقدير تباين الاختلاف في الأداء بين
المقدرات::

    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import KFold
    >>> from sklearn.model_selection import cross_val_score
    >>> import numpy as np

    >>> rng = np.random.RandomState(0)
    >>> X, y = make_classification(random_state=rng)
    >>> cv = KFold(shuffle=True, random_state=rng)
    >>> lda = LinearDiscriminantAnalysis()
    >>> nb = GaussianNB()

    >>> for est in (lda, nb):
    ...     print(cross_val_score(est, X, y, cv=cv))
    [0.8  0.75 0.75 0.7  0.85]
    [0.85 0.95 0.95 0.85 0.95]


مقارنة أداء
:class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` مقدر
مقابل :class:`~sklearn.naive_bayes.GaussianNB` مقدر **على كل طية** سيكون
خطأ: **الانقسامات التي يتم تقييم المقدرات عليها هي
مختلفة**. في الواقع، :func:`~sklearn.model_selection.cross_val_score` سوف
استدعاء `cv.split` داخليًا على نفس الشيء
:class:`~sklearn.model_selection.KFold` مثيل، لكن الانقسامات ستكون
مختلفة في كل مرة. هذا صحيح أيضًا بالنسبة لأي أداة تؤدي النموذج
الاختيار عبر التحقق المتبادل، على سبيل المثال
:class:`~sklearn.model_selection.GridSearchCV` و
:class:`~sklearn.model_selection.RandomizedSearchCV`: الدرجات ليست
قابلة للمقارنة من طية إلى أخرى عبر مكالمات مختلفة لـ `search.fit`، منذ ذلك الحين
تم استدعاء `cv.split` عدة مرات. ضمن مكالمة واحدة لـ
`search.fit`، ومع ذلك، فإن المقارنة من طية إلى أخرى ممكنة منذ البحث
المقدر يستدعي `cv.split` مرة واحدة فقط.

للحصول على نتائج قابلة للمقارنة من طية إلى أخرى في جميع السيناريوهات، يجب على المرء تمرير ملف
عدد صحيح لمقسم CV: `cv = KFold(shuffle=True، random_state=0)`.

.. note::
    بينما لا يُنصح بالمقارنة من طية إلى أخرى مع `RandomState`
    مثيلات، ومع ذلك، يمكن للمرء أن يتوقع أن تسمح متوسط الدرجات بالاستنتاج
    ما إذا كان أحد المقدرين أفضل من الآخر، طالما أن هناك ما يكفي من الطيات و
    يتم استخدام البيانات.

.. note::
    ما يهم في هذا المثال هو ما تم تمريره إلى
    :class:`~sklearn.model_selection.KFold`. سواء مررنا `RandomState`
    مثيل أو عدد صحيح لـ :func:`~sklearn.datasets.make_classification`
    غير ذي صلة بغرضنا التوضيحي. أيضا، لا
    :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` ولا
    :class:`~sklearn.naive_bayes.GaussianNB` مقدرات عشوائية.

التوصيات العامة
-----------------------

الحصول على نتائج قابلة للتكرار عبر عمليات تنفيذ متعددة
.......................................................

من أجل الحصول على نتائج قابلة للتكرار (أي ثابتة) عبر عمليات متعددة
* عمليات تنفيذ البرنامج *، نحتاج إلى إزالة جميع استخدامات `random_state=None`، والتي
هو الافتراضي. الطريقة الموصى بها هي إعلان متغير `rng` في الأعلى
من البرنامج، وقم بتمريره إلى أي كائن يقبل `random_state`
معامل::

    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from sklearn.datasets import make_classification
    >>> from sklearn.model_selection import train_test_split
    >>> import numpy as np

    >>> rng = np.random.RandomState(0)
    >>> X, y = make_classification(random_state=rng)
    >>> rf = RandomForestClassifier(random_state=rng)
    >>> X_train, X_test, y_train, y_test = train_test_split(X, y,
    ...                                                     random_state=rng)
    >>> rf.fit(X_train, y_train).score(X_test, y_test)
    0.84

نحن الآن مضمونون أن نتيجة هذا البرنامج النصي ستكون دائمًا 0.84، لا
بغض النظر عن عدد مرات تشغيله. تغيير متغير `rng` العام إلى
يجب أن تؤثر القيمة المختلفة على النتائج ، كما هو متوقع.

من الممكن أيضًا إعلان متغير `rng` كعدد صحيح. هذا قد
ومع ذلك، تؤدي إلى نتائج تحقق متبادل أقل قوة، كما سنرى في
القسم التالي.

.. note::
    لا نوصي بتعيين البذور `numpy` العالمية عن طريق الاتصال
    `np.random.seed(0)`. انظر `هنا
    <https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array/5837352#comment6712034_5837352>`_
    للحصول على مناقشة.

متانة نتائج التحقق المتبادل
......................................

عندما نقوم بتقييم أداء مقدر عشوائي عن طريق التحقق المتبادل، فإننا
نريد التأكد من أن المقدر يمكن أن ينتج تنبؤات دقيقة لجديد
البيانات، ولكننا نريد أيضًا التأكد من أن المقدر قوي فيما يتعلق به
تهيئة عشوائية. على سبيل المثال، نود الأوزان العشوائية
تهيئة :class:`~sklearn.linear_model.SGDClassifier` لتكون
جيد باستمرار عبر جميع الطيات: خلاف ذلك، عندما ندرب هذا المقدر
على بيانات جديدة، قد نكون غير محظوظين وقد تؤدي التهيئة العشوائية إلى
أداء سيء. وبالمثل، نريد أن تكون الغابة العشوائية قوية فيما يتعلق بـ
مجموعة من الميزات المحددة عشوائيًا التي ستستخدمها كل شجرة.

لهذه الأسباب، يُفضل تقييم التحقق المتبادل
الأداء من خلال ترك المقدر يستخدم RNG مختلفًا في كل طية. هذا
يتم ذلك عن طريق تمرير `RandomState` مثيل (أو `None`) إلى المقدر
تهيئة.

عندما نمرر عددًا صحيحًا، سيستخدم المقدر نفس RNG في كل طية:
إذا كان المقدر يعمل بشكل جيد (أو سيئ)، كما تم تقييمه بواسطة CV، فقد يكون ذلك فقط
لأننا كنا محظوظين (أو سيئ الحظ) مع تلك البذرة المحددة. تمرير المثيلات
يؤدي إلى نتائج CV أكثر قوة، ويجعل المقارنة بين مختلف
الخوارزميات أكثر عدالة. كما أنه يساعد في الحد من إغراء العلاج
مقدر RNG كمعامل فائق يمكن ضبطه.

ما إذا كنا نمرر `RandomState` مثيلات أو أعداد صحيحة إلى مقسمات CV ليس لديه
التأثير على المتانة، طالما يتم استدعاء `split` مرة واحدة فقط. عندما يكون `split`
يتم استدعاؤها عدة مرات، لم تعد المقارنة من طية إلى أخرى ممكنة. كما
نتيجة لذلك، عادةً ما يكون تمرير عدد صحيح إلى مقسمات CV أكثر أمانًا ويغطي معظم
حالات الاستخدام.

