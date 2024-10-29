.. _common_pitfalls:

=========================================
المزالق الشائعة والممارسات الموصى بها
=========================================

الغرض من هذا الفصل هو توضيح بعض المزالق الشائعة
والأنماط المضادة التي تحدث عند استخدام scikit-learn. يوفر
أمثلة على ما **لا** يجب القيام به، إلى جانب مثال صحيح
مناظر.

ما قبل المعالجة غير المتسقة
===============================

يوفر scikit-learn مكتبة من :ref:`data-transforms`، والتي
قد تنظف (راجع :ref:`preprocessing`)، تقلل
(راجع :ref:`data_reduction`)، توسع (راجع :ref:`kernel_approximation`)
أو توليد (راجع :ref:`feature_extraction`) تمثيلات الميزات.
إذا تم استخدام هذه التحولات على البيانات أثناء تدريب نموذج، فيجب أيضًا
استخدامها على مجموعات البيانات اللاحقة، سواء كانت بيانات الاختبار أو
البيانات في نظام الإنتاج. وإلا، فسيتم تغيير مساحة الميزة،
ولن يتمكن النموذج من الأداء بفعالية.

بالنسبة للمثال التالي، دعنا ننشئ مجموعة بيانات اصطناعية بميزة
واحدة::

    >>> from sklearn.datasets import make_regression
    >>> from sklearn.model_selection import train_test_split

    >>> random_state = 42
    >>> X, y = make_regression(random_state=random_state, n_features=1, noise=1)
    >>> X_train, X_test, y_train, y_test = train_test_split(
    ...     X, y, test_size=0.4, random_state=random_state)

**خطأ**

تمت معايرة مجموعة البيانات التدريبية، ولكن ليس مجموعة بيانات الاختبار، لذا فإن أداء النموذج
على مجموعة بيانات الاختبار أسوأ مما هو متوقع::

    >>> from sklearn.metrics import mean_squared_error
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.preprocessing import StandardScaler

    >>> scaler = StandardScaler()
    >>> X_train_transformed = scaler.fit_transform(X_train)
    >>> model = LinearRegression().fit(X_train_transformed, y_train)
    >>> mean_squared_error(y_test, model.predict(X_test))
    62.80...

**صحيح**

بدلاً من تمرير `X_test` غير المحول إلى `predict`، يجب علينا
تحويل بيانات الاختبار، بنفس الطريقة التي حولنا بها بيانات التدريب::

    >>> X_test_transformed = scaler.transform(X_test)
    >>> mean_squared_error(y_test, model.predict(X_test_transformed))
    0.90...

بدلاً من ذلك، نوصي باستخدام :class:`Pipeline
<sklearn.pipeline.Pipeline>`، مما يجعل من السهل تسلسل التحولات
مع المقدرات، ويقلل من احتمال نسيان التحول::

    >>> from sklearn.pipeline import make_pipeline

    >>> model = make_pipeline(StandardScaler(), LinearRegression())
    >>> model.fit(X_train, y_train)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('linearregression', LinearRegression())])
    >>> mean_squared_error(y_test, model.predict(X_test))
    0.90...

تساعد الأنابيب أيضًا في تجنب المزلق الشائع الآخر: تسرب بيانات الاختبار
إلى بيانات التدريب.

.. _data_leakage:

تسرب البيانات
============

يحدث تسرب البيانات عندما يتم استخدام معلومات لن تكون متاحة في وقت التنبؤ
عند بناء النموذج. يؤدي هذا إلى تقديرات أداء متفائلة للغاية، على سبيل المثال من :ref:`cross-validation
<cross_validation>`، وبالتالي أداء أسوأ عندما يتم استخدام النموذج
على بيانات جديدة بالفعل، على سبيل المثال أثناء الإنتاج.

أحد الأسباب الشائعة هو عدم الحفاظ على مجموعات البيانات الفرعية للاختبار والتدريب منفصلة.
يجب ألا تستخدم بيانات الاختبار مطلقًا لاتخاذ خيارات حول النموذج.
**القاعدة العامة هي عدم استدعاء** `fit` **على بيانات الاختبار**. على الرغم من أن هذا
قد يبدو واضحًا، إلا أنه من السهل تفويته في بعض الحالات، على سبيل المثال عند
تطبيق خطوات ما قبل المعالجة.

على الرغم من أنه يجب أن تتلقى كل من مجموعات البيانات الفرعية للتدريب والاختبار نفس
تحويل ما قبل المعالجة (كما هو موضح في القسم السابق)، فمن المهم
أن يتم تعلم هذه التحولات فقط من بيانات التدريب.
على سبيل المثال، إذا كان لديك
خطوة التطبيع حيث تقسم على القيمة المتوسطة، يجب أن يكون
المتوسط هو متوسط مجموعة البيانات الفرعية للتدريب، **ليس** متوسط جميع البيانات. إذا تم تضمين
مجموعة البيانات الفرعية للاختبار في حساب المتوسط، فإن المعلومات من مجموعة البيانات الفرعية للاختبار
تؤثر على النموذج.

كيفية تجنب تسرب البيانات
-------------------------

فيما يلي بعض النصائح لتجنب تسرب البيانات:

* قم دائمًا بتقسيم البيانات إلى مجموعات فرعية للتدريب والاختبار أولاً، خاصة
  قبل أي خطوات ما قبل المعالجة.
* لا تضمن بيانات الاختبار عند استخدام أساليب `fit` و `fit_transform`
  . يمكن أن يؤدي استخدام جميع البيانات، على سبيل المثال، `fit(X)`، إلى نتائج متفائلة للغاية
  الدرجات.

  على العكس من ذلك، يجب استخدام طريقة `transform` على كل من مجموعات البيانات الفرعية للتدريب والاختبار حيث يتم تطبيق نفس ما قبل المعالجة على جميع البيانات.
  يمكن تحقيق ذلك عن طريق استخدام `fit_transform` على مجموعة البيانات الفرعية للتدريب و
  `transform` على مجموعة البيانات الفرعية للاختبار.
* تعتبر أنابيب scikit-learn :ref:`pipeline <pipeline>` طريقة رائعة لمنع تسرب البيانات
  حيث تضمن أن يتم تنفيذ الطريقة المناسبة على مجموعة البيانات الفرعية الصحيحة. الأنبوب مثالي للاستخدام في
  وظائف التقاطع وضبط المعلمات.

يتم تفصيل مثال على تسرب البيانات أثناء ما قبل المعالجة أدناه.

تسرب البيانات أثناء ما قبل المعالجة
----------------------------------

.. note::
    نختار هنا توضيح تسرب البيانات بخطوة اختيار الميزة.
    ومع ذلك، فإن خطر التسرب هذا مهم مع جميع التحولات تقريبًا
    في scikit-learn، بما في ذلك (ولكن ليس على سبيل الحصر)
    :class:`~sklearn.preprocessing.StandardScaler`،
    :class:`~sklearn.impute.SimpleImputer`، و
    :class:`~sklearn.decomposition.PCA`.

يتوفر عدد من وظائف :ref:`feature_selection` في scikit-learn.
يمكنهم المساعدة في إزالة الميزات غير ذات الصلة والمتكررة والضجيج، وكذلك
تحسين وقت بناء النموذج وأدائه. كما هو الحال مع أي نوع آخر من
ما قبل المعالجة، يجب استخدام اختيار الميزة **فقط** باستخدام بيانات التدريب.
سيؤدي تضمين بيانات الاختبار في اختيار الميزة إلى تحيز نموذجك بشكل متفائل.

للتدليل، سنقوم بإنشاء مشكلة تصنيف ثنائي مع
10000 ميزة تم إنشاؤها عشوائيًا::

    >>> import numpy as np
    >>> n_samples, n_features, n_classes = 200, 10000, 2
    >>> rng = np.random.RandomState(42)
    >>> X = rng.standard_normal((n_samples, n_features))
    >>> y = rng.choice(n_classes, n_samples)

**خطأ**

يؤدي استخدام جميع البيانات لأداء اختيار الميزة إلى نتيجة دقة أعلى بكثير
من الفرصة، على الرغم من أن أهدافنا عشوائية تمامًا.
تعني هذه العشوائية أن `X` و `y` مستقلان، وبالتالي فإننا نتوقع
الدقة لتكون حوالي 0.5. ومع ذلك، نظرًا لأن خطوة اختيار الميزة
"ترى" بيانات الاختبار، فإن النموذج لديه ميزة غير عادلة. في المثال الخاطئ أدناه، نستخدم أولاً جميع البيانات لاختيار الميزة ثم
قم بتقسيم البيانات إلى مجموعات فرعية للتدريب والاختبار لتناسب النموذج. النتيجة هي
نتيجة دقة أعلى من المتوقع::

    >>> from sklearn.model_selection import train_test_split
    >>> from sklearn.feature_selection import SelectKBest
    >>> from sklearn.ensemble import GradientBoostingClassifier
    >>> from sklearn.metrics import accuracy_score

    >>> # ما قبل المعالجة غير الصحيحة: يتم تحويل البيانات بالكامل
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

لتجنب تسرب البيانات، من الجيد تقسيم بياناتك إلى مجموعات فرعية للتدريب
واختبار **أولاً**. يمكن بعد ذلك تشكيل اختيار الميزة باستخدام مجموعة بيانات التدريب فقط. لاحظ أنه كلما استخدمنا `fit` أو `fit_transform`، نستخدم فقط مجموعة بيانات التدريب. النتيجة هي ما نتوقعه للبيانات، قريب من الفرصة::

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

مرة أخرى، نوصي باستخدام :class:`~sklearn.pipeline.Pipeline` لربط
معًا مقدرات اختيار الميزة والنموذج. تضمن الأنابيب
أنه يتم استخدام بيانات التدريب فقط عند تنفيذ `fit` ويتم استخدام بيانات الاختبار
فقط لحساب نتيجة الدقة::

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

يمكن أيضًا تغذية الأنبوب في وظيفة التقاطع
مثل :func:`~sklearn.model_selection.cross_val_score`.
مرة أخرى، تضمن الأنابيب أنه يتم استخدام مجموعة البيانات الفرعية الصحيحة والمقدر
الطريقة المستخدمة أثناء التجهيز والتنبؤ::

    >>> from sklearn.model_selection import cross_val_score
    >>> scores = cross_val_score(pipeline, X, y)
    >>> print(f"Mean accuracy: {scores.mean():.2f}+/-{scores.std():.2f}")
    متوسط الدقة: 0.46+/-0.07


.. _randomness:

التحكم في العشوائية
======================

بعض كائنات scikit-learn عشوائية بطبيعتها. هذه عادة ما تكون المقدرات
(على سبيل المثال :class:`~sklearn.ensemble.RandomForestClassifier`) ومقسّمات التقاطع (على سبيل المثال :class:`~sklearn.model_selection.KFold`). يتم التحكم في العشوائية لهذه الكائنات عبر
معلمة `random_state` الخاصة بهم، كما هو موضح في
:term:`Glossary <random_state>`. يوسع هذا القسم إدخال المسرد
، ويصف الممارسات الجيدة والمزالق الشائعة فيما يتعلق بهذا
البارامتر الدقيق.

.. note:: ملخص التوصية

    للحصول على نتائج متينة مثالية للتقاطع (CV)، قم بتمرير
    `RandomState` عند إنشاء المقدرات، أو اترك `random_state`
    إلى `None`. يعد تمرير الأعداد الصحيحة إلى مقسّمات CV عادةً الخيار الأكثر أمانًا
    وهو مفضل؛ قد يكون تمرير `RandomState` إلى المقسّمات مفيدًا في بعض الأحيان لتحقيق حالات الاستخدام المحددة جدًا.
    بالنسبة لكل من المقدرات والمقسّمات، يؤدي تمرير عدد صحيح مقابل تمرير مثيل (أو `None`) إلى اختلافات دقيقة ولكنها مهمة،
    خاصة لإجراءات CV. هذه الاختلافات مهمة لفهمها عند الإبلاغ عن النتائج.

    للحصول على نتائج قابلة للتكرار عبر عمليات التنفيذ، قم بإزالة أي استخدام لـ
    `random_state=None`.

استخدام `None` أو `RandomState`، والمكالمات المتكررة لـ `fit` و `split`
--------------------------------------------------------------------------------

يحدد معلمة `random_state` ما إذا كانت المكالمات المتعددة لـ :term:`fit`
(للمقدرات) أو لـ :term:`split` (لمقسّمات CV) ستنتج نفس
النتائج، وفقًا لهذه القواعد:

- إذا تم تمرير عدد صحيح، فإن الاستدعاءات المتعددة لـ `fit` أو `split`
  دائمًا ما ينتج نفس النتائج.
- إذا تم تمرير `None` أو مثيل `RandomState`: `fit` و `split`
  ستؤدي إلى نتائج مختلفة في كل مرة يتم استدعاؤها، وسلسلة الاستدعاءات
  يستكشف جميع مصادر العشوائية. `None` هي القيمة الافتراضية لجميع
  معلمات `random_state`.

نوضح هنا هذه القواعد لكل من المقدرات ومقسّمات CV.

.. note::
    نظرًا لأن تمرير `random_state=None` يعادل تمرير مثيل `RandomState` العالمي
    من `numpy` (`random_state=np.random.mtrand._rand`)، فلن نذكر صراحةً
    `None` هنا. كل ما ينطبق على المثيلات ينطبق أيضًا على استخدام
    `None`.

المقدرات
..........

يعني تمرير المثيلات أن الاستدعاءات المتعددة لـ `fit` لن تنتج نفس
النتائج، حتى إذا تم تركيب المقدر على نفس البيانات وبنفس
المعلمات::

    >>> from sklearn.linear_model import SGDClassifier
    >>> from sklearn.datasets import make_classification
    >>> import numpy as np

    >>> rng = np.random.RandomState(42)
    >>> X, y = make_classification(n_features=5, random_state=rng)
    >>> sgd = SGDClassifier(random_state=rng)

    >>> sgd.fit(X, y).coef_
    array([[ 8.85418642,  4.79084103, -3.13077794,  8.11915045, -0.56479934]])

    >>> sgd.fit(X, y).coef_
    array([[ 6.70814003,  5.25291366, -7.55212743,  5.18197458,  1.37845099]])

يمكننا أن نرى من المقتطف أعلاه أن الاستدعاءات المتكررة لـ `sgd.fit`
أنتجت نماذج مختلفة، حتى إذا كانت البيانات هي نفسها. هذا لأن
تم استهلاك مولد الأرقام العشوائية (RNG) للمقدر (أي تم تغييره)
عندما يتم استدعاء `fit`، وسيتم استخدام هذا RNG المتحول في
المكالمات اللاحقة لـ `fit`. بالإضافة إلى ذلك، يتم مشاركة كائن `rng` عبر جميع الكائنات
التي تستخدمه، ونتيجة لذلك، تصبح هذه الكائنات مترابطة إلى حد ما. على سبيل المثال، سيؤثر المقدران اللذان يشتركان في نفس
مثيل `RandomState` على بعضهما البعض، كما سنرى لاحقًا عندما
نناقش الاستنساخ. هذه النقطة مهمة يجب مراعاتها عند تصحيح الأخطاء.

إذا كنا قد مررنا عددًا صحيحًا إلى معلمة `random_state` الخاصة بـ
:class:`~sklearn.linear_model.SGDClassifier`، لكنا قد حصلنا على نفس
النماذج، وبالتالي نفس الدرجات في كل مرة. عندما نمرر عددًا صحيحًا، يتم استخدام نفس RNG
عبر جميع المكالمات لـ `fit`. ما يحدث داخليًا هو أنه على الرغم من
يتم استهلاك RNG عند استدعاء `fit`، فإنه يتم دائمًا إعادة تعيينه إلى
حالته الأصلية في بداية `fit`.

مقسّمات CV
............

لدي مقسّمات CV العشوائية سلوك مشابه عند تمرير مثيل `RandomState`؛
استدعاء `split` عدة مرات ينتج عنه تقسيمات بيانات مختلفة::

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

يمكننا أن نرى أن التقسيمات مختلفة من المرة الثانية التي يتم فيها استدعاء `split`. قد يؤدي هذا إلى نتائج غير متوقعة إذا كنت تقارن أداء
المقدرات المتعددة عن طريق استدعاء `split` عدة مرات، كما سنرى في القسم التالي.

المزالق الشائعة والتعقيدات
------------------------------

على الرغم من أن القواعد التي تحكم معلمة `random_state` تبدو بسيطة،
إلا أن لها بعض الآثار الدقيقة. في بعض الحالات، يمكن أن يؤدي ذلك حتى إلى
استنتاجات خاطئة.

المقدرات
..........

**أنواع `random_state` المختلفة تؤدي إلى إجراءات تقاطع مختلفة**

وفقًا لنوع معلمة `random_state`، سيتصرف المقدرات بشكل مختلف، خاصة في إجراءات التقاطع. ضع في اعتبارك
المقتطف التالي::

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

نرى أن نتائج التقاطع الصحيحة لـ `rf_123` و `rf_inst`
مختلفة، كما هو متوقع حيث لم نمرر نفس معلمة `random_state`
. ومع ذلك، فإن الاختلاف بين هذه الدرجات أكثر دقة مما يبدو، و **إجراءات التقاطع التي قام بها**
:func:`~sklearn.model_selection.cross_val_score` **تختلف بشكل كبير في كل حالة**:

- نظرًا لأنه تم تمرير عدد صحيح إلى `rf_123`، فإن كل استدعاء لـ `fit` يستخدم نفس RNG:
  هذا يعني أن جميع الخصائص العشوائية لمقدر الغابة العشوائية
  ستكون هي نفسها لكل من الطيات الخمس لإجراء CV. على وجه الخصوص،
  سيكون الجزء الفرعي (المختار عشوائيًا) من الميزات للمقدر هو نفسه عبر جميع الطيات.
- نظرًا لأنه تم تمرير مثيل `RandomState` إلى `rf_inst`، فإن كل استدعاء لـ `fit`
  يبدأ من RNG مختلف. ونتيجة لذلك، فإن المجموعة الفرعية العشوائية من الميزات
  ستكون مختلفة لكل الطيات.

على الرغم من أن وجود RNG ثابت عبر الطيات ليس خاطئًا في حد ذاته، فإننا عادة ما نريد
نتائج CV متينة فيما يتعلق بعشوائية المقدر. ونتيجة لذلك، قد يكون تمرير مثيل بدلاً من عدد صحيح
أفضل، حيث سيسمح لـ RNG المقدر بالاختلاف لكل طية.

.. note::
    هنا، سيستخدم :func:`~sklearn.model_selection.cross_val_score` مقسّم CV غير عشوائي (كما هو الافتراضي)، لذا سيتم تقييم كل من المقدرات
    على نفس التقسيمات. هذا القسم ليس حول التباين في التقسيمات. أيضًا، سواء مررنا عددًا صحيحًا أو مثيلاً إلى
    :func:`~sklearn.datasets.make_classification` ليس ذا صلة بغرض التوضيح لدينا: ما يهم هو ما نمرره إلى
    المقدر :class:`~sklearn.ensemble.RandomForestClassifier`.

.. dropdown:: الاستنساخ

    تأثير جانبي دقيق آخر لتمرير مثيلات `RandomState` هو كيفية عمل
    :func:`~sklearn.base.clone`::

        >>> from sklearn import clone
        >>> from sklearn.ensemble import RandomForestClassifier
        >>> import numpy as np

        >>> rng = np.random.RandomState(0)
        >>> a = RandomForestClassifier(random_state=rng)
        >>> b = clone(a)

    نظرًا لأنه تم تمرير مثيل `RandomState` إلى `a`، فإن `a` و `b` ليسا استنساخات
    بالمعنى الدقيق للكلمة، ولكن استنساخات بالمعنى الإحصائي: `a` و `b`
    ستظل نماذج مختلفة، حتى عند استدعاء `fit(X, y)` على نفس
    البيانات. علاوة على ذلك، فإن `a` و `b` سيؤثران على بعضهما البعض حيث يتشاركان في نفس
    RNG الداخلي: سيؤدي استدعاء `a.fit` إلى استهلاك RNG الخاص بـ `b`، وسيؤدي استدعاء
    `b.fit` إلى استهلاك RNG الخاص بـ `a`، نظرًا لأنهما متطابقان. هذه النقطة صحيحة لأي مقدرات تشترك في
    معلمة `random_state`؛ ليس خاصًا بالاستنساخ.

    إذا تم تمرير عدد صحيح، فستكون `a` و `b` استنساخات دقيقة ولن يؤثرا على بعضهما البعض.

    .. warning::
        على الرغم من أن :func:`~sklearn.base.clone` نادرًا ما يتم استخدامه في كود المستخدم، إلا أنه يتم استدعاؤه بشكل مستمر في جميع أنحاء شفرة scikit-learn: على وجه الخصوص، معظم
        المقدرات الميتا التي تقبل المقدرات غير المجهزة تستدعي
        :func:`~sklearn.base.clone` داخليًا
        (:class:`~sklearn.model_selection.GridSearchCV`،
        :class:`~sklearn.ensemble.StackingClassifier`،
        :class:`~sklearn.calibration.CalibratedClassifierCV`، إلخ).


مقسّمات CV
............

عند تمرير مثيل `RandomState`، تنتج مقسّمات CV تقسيمات مختلفة
في كل مرة يتم فيها استدعاء `split`. عند مقارنة المقدرات المختلفة، يمكن أن يؤدي ذلك إلى
المبالغة في تقدير تباين الفرق في الأداء بين
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


سيكون إجراء مقارنة مباشرة لأداء
:class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis`
مقابل المقدر :class:`~sklearn.naive_bayes.GaussianNB` **على كل طية** خطأً: **التقسيمات التي يتم تقييم المقدرات عليها
مختلفة**. في الواقع، سيقوم :func:`~sklearn.model_selection.cross_val_score`
باستدعاء `cv.split` على نفس
مثيل :class:`~sklearn.model_selection.KFold`، ولكن ستكون التقسيمات
مختلفة في كل مرة. هذا صحيح أيضًا لأي أداة تقوم باختيار النموذج عبر التقاطع، على سبيل المثال
:class:`~sklearn.model_selection.GridSearchCV` و
:class:`~sklearn.model_selection.RandomizedSearchCV`: الدرجات غير قابلة للمقارنة من طية إلى طية عبر مكالمات مختلفة لـ `search.fit`،
نظرًا لأنه تم استدعاء `cv.split` عدة مرات. داخل مكالمة واحدة لـ
`search.fit`، ومع ذلك، تكون المقارنة من طية إلى طية ممكنة حيث يقوم المقدر
بالبحث فقط عن استدعاء `cv.split` مرة واحدة.

للحصول على نتائج قابلة للمقارنة من طية إلى طية في جميع السيناريوهات، يجب تمرير عدد صحيح إلى
مقسّم CV: `cv = KFold(shuffle=True, random_state=0)`.

.. note::
    على الرغم من أن المقارنة من طية إلى طية غير ممكنة مع مثيلات `RandomState`،
    يمكن للمرء مع ذلك توقع أن تسمح الدرجات المتوسطة باستنتاج ما إذا كان أحد المقدرات أفضل من الآخر، طالما تم استخدام عدد كافٍ من الطيات والبيانات.

.. note::
    ما يهم في هذا المثال هو ما تم تمريره إلى
    :class:`~sklearn.model_selection.KFold`. سواء مررنا مثيل `RandomState` أو عددًا صحيحًا إلى
    :func:`~sklearn.datasets.make_classification` ليس ذا صلة بغرض التوضيح لدينا. أيضًا، لا
    :class:`~sklearn.discriminant_analysis.LinearDiscriminantAnalysis` ولا
    :class:`~sklearn.naive_bayes.GaussianNB` هي مقدرات عشوائية.

توصيات عامة
-----------------------

الحصول على نتائج قابلة للتكرار عبر عمليات التنفيذ المتعددة
.......................................................

من أجل الحصول على نتائج قابلة للتكرار (أي ثابتة) عبر عمليات التنفيذ المتعددة
*تنفيذ البرنامج*، نحتاج إلى إزالة جميع استخدامات `random_state=None`، والتي
هو الافتراضي. الطريقة الموصى بها هي إعلان متغير `rng` في أعلى
البرنامج، ومرره إلى أي كائن يقبل معلمة `random_state`::

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

نحن الآن مضمونون أن نتيجة هذا البرنامج النصي ستكون دائمًا 0.84، بغض النظر عن عدد المرات التي ننفذها. يجب أن يؤثر تغيير متغير `rng` العالمي على النتائج، كما هو متوقع.

من الممكن أيضًا إعلان متغير `rng` كعدد صحيح. ومع ذلك، قد يؤدي هذا إلى نتائج أقل متانة للتقاطع، كما سنرى في القسم التالي.

.. note::
    لا نوصي بتعيين البذرة العالمية لـ `numpy` عن طريق استدعاء
    `np.random.seed(0)`. انظر `هنا
    <https://stackoverflow.com/questions/5836335/consistently-create-same-random-numpy-array/5837352#comment6712034_5837352>`_
    للمناقشة.

متانة نتائج التقاطع
......................................

عندما نقيم أداء المقدر العشوائي عن طريق التقاطع، نريد
تأكد من أن المقدر يمكنه تقديم تنبؤات دقيقة للبيانات الجديدة، ولكننا نريد أيضًا
تأكد من أن المقدر متين فيما يتعلق بتهيئته العشوائية. على سبيل المثال، نود أن تكون تهيئة الأوزان العشوائية لـ
:class:`~sklearn.linear_model.SGDClassifier`
متسقة عبر جميع الطيات: وإلا، عندما نقوم بتدريب هذا المقدر
على بيانات جديدة، قد نحصل على حظ سيئ وقد تؤدي تهيئة عشوائية إلى
أداء سيئ. وبالمثل، نريد أن تكون الغابة العشوائية متينة فيما يتعلق بمجموعة الميزات العشوائية التي سيستخدمها كل شجرة.

لهذه الأسباب، من الأفضل تقييم أداء التقاطع عن طريق السماح للمقدر باستخدام
RNG مختلف على كل طية. يتم ذلك عن طريق تمرير مثيل `RandomState` (أو `None`) إلى
تهيئة المقدر.

عندما نمرر عددًا صحيحًا، سيستخدم المقدر نفس RNG على كل طية:
إذا كان أداء المقدر جيدًا (أو سيئًا)، كما هو مقدر من CV، فقد يكون ذلك لأننا
حصلنا على حظ سعيد (أو سيئ) مع تلك البذرة المحددة. يؤدي تمرير المثيلات إلى نتائج أكثر متانة لـ CV، ويجعل
مقارنة الخوارزميات المختلفة أكثر عدلاً. كما أنه يساعد في الحد من إغراء معاملة
RNG المقدر كمعلمة يمكن ضبطها.

سواء قمنا بتمرير مثيلات `RandomState` أو أعداد صحيحة إلى مقسّمات CV ليس له تأثير على المتانة، طالما تم استدعاء `split` مرة واحدة. عندما يتم استدعاء `split` عدة مرات، لم تعد المقارنة من طية إلى طية ممكنة. ونتيجة لذلك، فإن تمرير عدد صحيح إلى مقسّمات CV هو عادةً الخيار الأكثر أمانًا ويغطي معظم حالات الاستخدام.