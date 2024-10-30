
البدء
========

الغرض من هذا الدليل هو توضيح بعض الميزات الرئيسية التي
يوفرها ``scikit-learn``. يفترض معرفة عمل أساسية جدًا
بممارسات التعلم الآلي (ملاءمة النموذج، والتنبؤ، والتحقق المتبادل،
إلخ). يرجى الرجوع إلى :ref:`تعليمات التثبيت
<installation-instructions>` لتثبيت ``scikit-learn``.

``Scikit-learn`` هي مكتبة تعلم آلي مفتوحة المصدر تدعم
التعلم الخاضع للإشراف وغير الخاضع للإشراف. كما يوفر أدوات مختلفة
لملاءمة النموذج، ومعالجة البيانات الأولية، واختيار النموذج، وتقييم النموذج،
والعديد من الأدوات المساعدة الأخرى.

الملاءمة والتنبؤ: أساسيات المقدر
-----------------------------------

يوفر ``Scikit-learn`` العشرات من خوارزميات ونماذج التعلم الآلي المضمنة،
تسمى :term:`المقدرات`. يمكن ملاءمة كل مقدر مع بعض البيانات
باستخدام :term:`fit` الطريقة.

فيما يلي مثال بسيط حيث نقوم بملاءمة
:class:`~sklearn.ensemble.RandomForestClassifier` لبعض البيانات الأساسية جدًا::

  >>> from sklearn.ensemble import RandomForestClassifier
  >>> clf = RandomForestClassifier(random_state=0)
  >>> X = [[ 1,  2,  3],  # 2 عينات، 3 ميزات
  ...      [11, 12, 13]]
  >>> y = [0, 1]  # فئات كل عينة
  >>> clf.fit(X, y)
  RandomForestClassifier(random_state=0)

تقبل طريقة :term:`fit` عمومًا مدخلين:

- مصفوفة العينات (أو مصفوفة التصميم) :term:`X`. حجم ``X``
  هو عادةً ``(n_samples، n_features)``، مما يعني أن العينات
  يتم تمثيلها كصفوف ويتم تمثيل الميزات كأعمدة.
- قيم الهدف :term:`y` وهي أرقام حقيقية لمهام الانحدار، أو
  أعداد صحيحة للتصنيف (أو أي مجموعة أخرى منفصلة من القيم). لـ
  مهام التعلم غير الخاضعة للإشراف، لا يلزم تحديد ``y``. ``y`` هو
  عادةً مصفوفة 1d حيث يتوافق الإدخال ``i`` مع هدف
  عينة ``i`` (صف) من ``X``.

من المتوقع أن يكون كل من ``X`` و ``y`` مصفوفات numpy أو ما يعادلها
:term:`array-like` أنواع البيانات، على الرغم من أن بعض المقدرات تعمل مع أخرى
التنسيقات مثل المصفوفات المتفرقة.

بمجرد ملاءمة المقدر، يمكن استخدامه للتنبؤ بقيم الهدف
من بيانات جديدة. لست بحاجة إلى إعادة تدريب المقدر::

  >>> clf.predict(X)  # توقع فئات بيانات التدريب
  array([0, 1])
  >>> clf.predict([[4, 5, 6], [14, 15, 16]])  # توقع فئات البيانات الجديدة
  array([0, 1])

يمكنك التحقق من :ref:`ml_map` حول كيفية اختيار النموذج المناسب لحالة الاستخدام الخاصة بك.

المحولات والمعالجات المسبقة
-------------------------------

غالبًا ما تتكون سير عمل التعلم الآلي من أجزاء مختلفة. نموذجي
يتكون خط الأنابيب من خطوة معالجة مسبقة تقوم بتحويل أو إدخال
البيانات، ومتنبئ نهائي يتنبأ بقيم الهدف.

في ``scikit-learn``، تتبع المعالجات المسبقة والمحولات نفس واجهة برمجة التطبيقات
مثل كائنات المقدر (فهي في الواقع ترث كلها من نفس الشيء
``BaseEstimator`` class). كائنات المحول ليس لها
:term:`predict` طريقة بل بالأحرى :term:`transform` طريقة تخرج
مصفوفة عينة محولة حديثًا ``X``::

  >>> from sklearn.preprocessing import StandardScaler
  >>> X = [[0, 15],
  ...      [1, -10]]
  >>> # قياس البيانات وفقًا لقيم القياس المحسوبة
  >>> StandardScaler().fit(X).transform(X)
  array([[-1.,  1.],
         [ 1., -1.]])

في بعض الأحيان، تريد تطبيق تحويلات مختلفة على ميزات مختلفة:
:ref:`ColumnTransformer<column_transformer>` مصمم لهؤلاء
حالات الاستخدام.

خطوط الأنابيب: سلسلة المعالجات المسبقة والمقدرات
--------------------------------------------------

يمكن دمج المحولات والمقدرات (المتنبئون) معًا في
كائن توحيد واحد: :class:`~sklearn.pipeline.Pipeline`. يقدم خط 
الأنابيب
نفس واجهة برمجة التطبيقات مثل المقدر العادي: يمكن تركيبه واستخدامه
للتنبؤ بـ ``fit`` و ``predict``. كما سنرى لاحقًا، باستخدام
سيمنعك خط الأنابيب أيضًا من تسرب البيانات، أي الكشف عن بعض
اختبار البيانات في بيانات التدريب الخاصة بك.

في المثال التالي، :ref:`نقوم بتحميل مجموعة بيانات Iris <datasets>`، ونقسمها
إلى مجموعات التدريب والاختبار، وحساب درجة دقة خط الأنابيب على
بيانات الاختبار::

  >>> from sklearn.preprocessing import StandardScaler
  >>> from sklearn.linear_model import LogisticRegression
  >>> from sklearn.pipeline import make_pipeline
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.model_selection import train_test_split
  >>> from sklearn.metrics import accuracy_score
  ...
  >>> # إنشاء كائن خط أنابيب
  >>> pipe = make_pipeline(
  ...     StandardScaler(),
  ...     LogisticRegression()
  ... )
  ...
  >>> # تحميل مجموعة بيانات Iris وتقسيمها إلى مجموعات التدريب والاختبار
  >>> X, y = load_iris(return_X_y=True)
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  ...
  >>> # ملاءمة خط الأنابيب بأكمله
  >>> pipe.fit(X_train, y_train)
  Pipeline(steps=[('standardscaler', StandardScaler()),
                  ('logisticregression', LogisticRegression())])
  >>> # يمكننا الآن استخدامه مثل أي مقدر آخر
  >>> accuracy_score(pipe.predict(X_test), y_test)
  0.97...

تقييم النموذج
----------------

لا تستلزم ملاءمة نموذج مع بعض البيانات أنه سيتنبأ جيدًا
على بيانات غير مرئية. هذا يحتاج إلى تقييم مباشر. لقد رأينا للتو
:func:`~sklearn.model_selection.train_test_split` مساعد يقسم
مجموعة بيانات إلى مجموعات تدريب واختبار، لكن ``scikit-learn`` يوفر العديد من
أدوات أخرى لتقييم النموذج، خاصة لـ :ref:`التحقق المتبادل
<cross_validation>`.

نعرض هنا بإيجاز كيفية إجراء إجراء تحقق متبادل من 5 طيات،
باستخدام :func:`~sklearn.model_selection.cross_validate` مساعد. لاحظ ذلك
من الممكن أيضًا التكرار يدويًا عبر الطيات، واستخدام مختلفة
استراتيجيات تقسيم البيانات، واستخدام وظائف التسجيل المخصصة. يرجى الرجوع إلى
:ref:`دليل المستخدم <cross_validation>` لمزيد من التفاصيل::

  >>> from sklearn.datasets import make_regression
  >>> from sklearn.linear_model import LinearRegression
  >>> from sklearn.model_selection import cross_validate
  ...
  >>> X, y = make_regression(n_samples=1000, random_state=0)
  >>> lr = LinearRegression()
  ...
  >>> result = cross_validate(lr, X, y)  # الافتراضي هو 5 أضعاف CV
  >>> result['test_score']  # درجة r_squared عالية لأن مجموعة البيانات سهلة
  array([1., 1., 1., 1., 1.])

عمليات البحث التلقائي عن المعلمات
--------------------------------------

جميع المقدرات لها معلمات (غالبًا ما تسمى المعلمات الفائقة في
الأدب) التي يمكن ضبطها. قوة التعميم لمقدر
غالبًا ما يعتمد بشكل حاسم على بعض المعلمات. على سبيل المثال
:class:`~sklearn.ensemble.RandomForestRegressor` لديه ``n_estimators``
معلمة تحدد عدد الأشجار في الغابة، و
معلمة ``max_depth`` التي تحدد أقصى عمق لكل شجرة.
في كثير من الأحيان، ليس من الواضح ما هي القيم الدقيقة لهذه المعلمات
يجب أن يكون لأنها تعتمد على البيانات الموجودة.

يوفر ``Scikit-learn`` أدوات للعثور تلقائيًا على أفضل معلمة
مجموعات (عبر التحقق المتبادل). في المثال التالي، نحن عشوائيا
البحث عبر مساحة المعلمة لغابة عشوائية مع
:class:`~sklearn.model_selection.RandomizedSearchCV` كائن. عندما يكون البحث
انتهى، :class:`~sklearn.model_selection.RandomizedSearchCV` يتصرف مثل
:class:`~sklearn.ensemble.RandomForestRegressor` التي تم تركيبها
مع أفضل مجموعة من المعلمات. اقرأ المزيد في :ref:`دليل المستخدم
<grid_search>`::

  >>> from sklearn.datasets import fetch_california_housing
  >>> from sklearn.ensemble import RandomForestRegressor
  >>> from sklearn.model_selection import RandomizedSearchCV
  >>> from sklearn.model_selection import train_test_split
  >>> from scipy.stats import randint
  ...
  >>> X, y = fetch_california_housing(return_X_y=True)
  >>> X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
  ...
  >>> # تحديد مساحة المعلمة التي سيتم البحث عنها
  >>> param_distributions = {'n_estimators': randint(1, 5),
  ...                        'max_depth': randint(5, 10)}
  ...
  >>> # الآن قم بإنشاء كائن searchCV وقم بملاءمته مع البيانات
  >>> search = RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0),
  ...                             n_iter=5,
  ...                             param_distributions=param_distributions,
  ...                             random_state=0)
  >>> search.fit(X_train, y_train)
  RandomizedSearchCV(estimator=RandomForestRegressor(random_state=0), n_iter=5,
                     param_distributions={'max_depth': ...,
                                          'n_estimators': ...},
                     random_state=0)
  >>> search.best_params_
  {'max_depth': 9, 'n_estimators': 4}

  >>> # يعمل كائن البحث الآن مثل مقدر عادي للغابات العشوائية
  >>> # مع max_depth=9 و n_estimators=4
  >>> search.score(X_test, y_test)
  0.73...

.. note::

    في الممارسة العملية، تريد دائمًا :ref:`البحث عبر خط أنابيب
    <composite_grid_search>`، بدلاً من مقدر واحد. أحد الأسباب الرئيسية
    هو أنه إذا قمت بتطبيق خطوة معالجة مسبقة على مجموعة البيانات بأكملها
    دون استخدام خط أنابيب، ثم إجراء أي نوع من التحقق المتبادل،
    ستكون قد خرقت الافتراض الأساسي للاستقلال بين
    بيانات التدريب والاختبار. في الواقع، منذ أن قمت بمعالجة البيانات مسبقًا
    باستخدام مجموعة البيانات بأكملها، بعض المعلومات حول مجموعات الاختبار
    متاحة لمجموعات التدريب. سيؤدي هذا إلى المبالغة في تقدير
    قوة التعميم للمقدر (يمكنك قراءة المزيد في `منشور Kaggle هذا
    <https://www.kaggle.com/alexisbcook/data-leakage>`_).

    سيبقيك استخدام خط الأنابيب للتحقق المتبادل والبحث إلى حد كبير
    من هذا الفخ الشائع.


الخطوات التالية
-----------------

لقد غطينا بإيجاز ملاءمة المقدر والتنبؤ، والمعالجة المسبقة
الخطوات، وخطوط الأنابيب، وأدوات التحقق المتبادل، والبارامترات التلقائية
عمليات البحث. يجب أن يمنحك هذا الدليل نظرة عامة على بعض
الميزات الرئيسية للمكتبة، ولكن هناك ما هو أكثر من
``scikit-learn``!

يرجى الرجوع إلى :ref:`user_guide` للحصول على تفاصيل حول جميع الأدوات التي
نحن نقدم. يمكنك أيضًا العثور على قائمة شاملة بواجهة برمجة التطبيقات العامة في
:ref:`api_ref`.

يمكنك أيضًا إلقاء نظرة على العديد من :ref:`الأمثلة <general_examples>` التي
توضيح استخدام ``scikit-learn`` في العديد من السياقات المختلفة.
