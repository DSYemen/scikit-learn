
.. currentmodule:: sklearn.feature_selection

.. _feature_selection:

=================
اختيار الميزات
=================


يمكن استخدام الفئات في وحدة :mod:`sklearn.feature_selection` لـ
اختيار الميزات / تقليل الأبعاد على مجموعات العينات، إما لـ
تحسين درجات دقة المُقدِّرات أو لتعزيز أدائها على مجموعات البيانات
عالية الأبعاد للغاية.


.. _variance_threshold:

إزالة الميزات ذات التباين المنخفض
===================================

:class:`VarianceThreshold` هو نهج أساسي بسيط لاختيار الميزات.
إنه يُزيل جميع الميزات التي لا يُلبي تباينها عتبة مُعينة.
افتراضيًا، يُزيل جميع الميزات ذات التباين الصفري،
أي الميزات التي لها نفس القيمة في جميع العينات.

على سبيل المثال، لنفترض أن لدينا مجموعة بيانات ذات ميزات منطقية،
ونُريد إزالة جميع الميزات التي تكون إما واحدًا أو صفرًا (مُشغَّلة أو مُطفأة)
في أكثر من 80% من العينات.
الميزات المنطقية هي متغيرات عشوائية برنولي،
ويتم إعطاء تباين هذه المتغيرات بواسطة

.. math:: \mathrm{Var}[X] = p(1 - p)

حتى نتمكن من الاختيار باستخدام العتبة ``.8 * (1 - .8)``::

  >>> from sklearn.feature_selection import VarianceThreshold
  >>> X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
  >>> sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
  >>> sel.fit_transform(X)
  array([[0, 1],
         [1, 0],
         [0, 0],
         [1, 1],
         [1, 0],
         [1, 1]])

كما هو مُتوقع، أزال ``VarianceThreshold`` العمود الأول،
الذي يحتوي على احتمال :math:`p = 5/6 > .8` لاحتواء صفر.

.. _univariate_feature_selection:

اختيار الميزات أحادية المتغير
============================

يعمل اختيار الميزات أحادية المتغير عن طريق اختيار أفضل الميزات بناءً على
الاختبارات الإحصائية أحادية المتغير. يمكن اعتباره خطوة مُعالجة مُسبقة
لمُقدِّر. يُظهِر Scikit-learn إجراءات اختيار الميزات
ككائنات تُطبق أسلوب ``transform``:

* :class:`SelectKBest` يُزيل كل شيء باستثناء :math:`k` أعلى الميزات
  درجة

* :class:`SelectPercentile` يُزيل كل شيء باستثناء أعلى نسبة مُحددة من قبل
  المستخدم من الميزات درجة

* استخدام الاختبارات الإحصائية أحادية المتغير الشائعة لكل ميزة:
  مُعدل الإيجابيات الخاطئة :class:`SelectFpr`، مُعدل الاكتشاف الخاطئ
  :class:`SelectFdr`، أو خطأ العائلة الحكيم :class:`SelectFwe`.

* :class:`GenericUnivariateSelect` يسمح بإجراء اختيار ميزات
  أحادي المتغير باستخدام إستراتيجية قابلة للتكوين. هذا يسمح باختيار أفضل
  إستراتيجية اختيار أحادي المتغير باستخدام مُقدِّر بحث المعلمات
  الفائقة.

على سبيل المثال، يمكننا استخدام اختبار F لاسترداد الاثنين
من أفضل الميزات لمجموعة بيانات على النحو التالي:

  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectKBest
  >>> from sklearn.feature_selection import f_classif
  >>> X, y = load_iris(return_X_y=True)
  >>> X.shape
  (150, 4)
  >>> X_new = SelectKBest(f_classif, k=2).fit_transform(X, y)
  >>> X_new.shape
  (150, 2)

تأخذ هذه الكائنات كمدخلات دالة تسجيل تُعيد درجات أحادية المتغير
وقيم p (أو درجات فقط لـ :class:`SelectKBest` و
:class:`SelectPercentile`):

* للانحدار: :func:`r_regression` و :func:`f_regression` و :func:`mutual_info_regression`

* للتصنيف: :func:`chi2` و :func:`f_classif` و :func:`mutual_info_classif`

تُقدِّر الأساليب القائمة على اختبار F درجة التبعية الخطية بين
متغيرين عشوائيين. من ناحية أخرى، يمكن لأساليب المعلومات المتبادلة التقاط
أي نوع من التبعية الإحصائية، ولكن نظرًا لكونها غير بارامترية، فإنها تتطلب المزيد
من العينات للتقدير الدقيق. لاحظ أنه يجب تطبيق اختبار :math:`\chi^2` فقط
على الميزات غير السالبة، مثل الترددات.

.. topic:: اختيار الميزات مع البيانات المتفرقة

   إذا كنت تستخدم بيانات متفرقة (أي البيانات المُمثلة كمصفوفات متفرقة)،
   :func:`chi2` و :func:`mutual_info_regression` و :func:`mutual_info_classif`
   ستتعامل مع البيانات دون جعلها كثيفة.

.. warning::

    احذر من عدم استخدام دالة تسجيل انحدار مع مشكلة
    تصنيف، ستحصل على نتائج غير مُجدية.

.. note::

    يدعم :class:`SelectPercentile` و :class:`SelectKBest` اختيار الميزات غير
    الخاضع للإشراف أيضًا. يحتاج المرء إلى توفير `score_func` حيث `y = None`.
    يجب أن تستخدم `score_func` داخليًا `X` لحساب الدرجات.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_feature_selection.py`

* :ref:`sphx_glr_auto_examples_feature_selection_plot_f_test_vs_mi.py`

.. _rfe:

إزالة الميزات التكراري
=============================

بالنظر إلى مُقدِّر خارجي يُعيِّن أوزانًا للميزات (على سبيل المثال،
معاملات نموذج خطي)، فإن هدف إزالة الميزات التكراري (:class:`RFE`)
هو تحديد الميزات عن طريق النظر بشكل متكرر في مجموعات أصغر وأصغر من
الميزات. أولاً، يتم تدريب المُقدِّر على المجموعة الأولية من الميزات و
يتم الحصول على أهمية كل ميزة إما من خلال أي سمة مُحددة
(مثل ``coef_`` أو ``feature_importances_``) أو دالة قابلة للاستدعاء. ثم، يتم
تشذيب أقل الميزات أهمية من المجموعة الحالية من الميزات. يتم تكرار هذا الإجراء
بشكل متكرر على المجموعة المُشذَّبة حتى يتم الوصول إلى العدد المطلوب من الميزات
التي سيتم تحديدها في النهاية.

:class:`RFECV` يُجري RFE في حلقة تحقق متبادل للعثور على
العدد الأمثل للميزات. بمزيد من التفصيل، يتم ضبط عدد الميزات المحددة
تلقائيًا عن طريق ملاءمة مُحدد :class:`RFE` على تقسيمات
التحقق المتبادل المختلفة (التي تُوفرها معلمة `cv`). يتم تقييم أداء
مُحدد :class:`RFE` باستخدام `scorer` لعدد مختلف من
الميزات المحددة ويتم تجميعها معًا. أخيرًا، يتم حساب متوسط الدرجات
عبر الطيات ويتم تعيين عدد الميزات المحددة على عدد
الميزات التي تُعظِّم درجة التحقق المتبادل.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_digits.py`: مثال على إزالة الميزات
  التكراري يُظهر أهمية وحدات البكسل في مهمة تصنيف الأرقام.

* :ref:`sphx_glr_auto_examples_feature_selection_plot_rfe_with_cross_validation.py`:
  مثال على إزالة الميزات التكراري مع الضبط التلقائي لعدد
  الميزات المحددة مع التحقق المتبادل.

.. _select_from_model:

اختيار الميزات باستخدام SelectFromModel
=======================================

:class:`SelectFromModel` هو مُحوِّل تعريف يمكن استخدامه جنبًا إلى جنب مع أي
مُقدِّر يُعيِّن أهمية لكل ميزة من خلال سمة مُحددة (مثل
``coef_`` أو ``feature_importances_``) أو عبر دالة `importance_getter` قابلة للاستدعاء
بعد الملاءمة.
تُعتبر الميزات غير مهمة ويتم إزالتها إذا كانت
أهمية قيم الميزات المُقابلة أقل من
معلمة ``threshold`` المُقدمة. بصرف النظر عن تحديد العتبة عدديًا،
هناك استدلالات مُدمجة للعثور على عتبة باستخدام وسيطة سلسلة.
الاستدلالات المتاحة هي "mean" و "median" ومُضاعفات عائمة لهذه مثل
"0.1 * mean". بالاقتران مع معيار `threshold`، يمكن للمرء استخدام
معلمة `max_features` لتعيين حد لعدد الميزات المراد تحديدها.

للحصول على أمثلة حول كيفية استخدامها، ارجع إلى الأقسام أدناه.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_select_from_model_diabetes.py`

.. _l1_feature_selection:

اختيار الميزات القائم على L1
--------------------------

.. currentmodule:: sklearn

:ref:`النماذج الخطية <linear_model>` المُعاقبة بقاعدة L1 لها
حلول متفرقة: العديد من معاملاتها المُقدَّرة تساوي صفرًا. عندما يكون الهدف
هو تقليل أبعاد البيانات لاستخدامها مع مُصنف آخر،
يمكن استخدامها مع :class:`~feature_selection.SelectFromModel`
لاختيار المعاملات غير الصفرية. على وجه الخصوص، المُقدِّرات المتفرقة المفيدة
لهذا الغرض هي :class:`~linear_model.Lasso` للانحدار، و
:class:`~linear_model.LogisticRegression` و :class:`~svm.LinearSVC`
للتصنيف::

  >>> from sklearn.svm import LinearSVC
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> X, y = load_iris(return_X_y=True)
  >>> X.shape
  (150, 4)
  >>> lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
  >>> model = SelectFromModel(lsvc, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape
  (150, 3)

مع SVMs والانحدار اللوجستي، تتحكم المعلمة C في التفرق:
كلما صغرت C، قل عدد الميزات المحددة. مع Lasso، كلما ارتفعت
معلمة alpha، قل عدد الميزات المحددة.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_linear_model_plot_lasso_dense_vs_sparse_data.py`.

.. _compressive_sensing:

.. dropdown:: استرداد L1 والاستشعار المضغوط

  لاختيار جيد لـ alpha، يمكن لـ :ref:`lasso` استرداد
  مجموعة المتغيرات غير الصفرية الدقيقة باستخدام عدد قليل من الملاحظات، بشرط
  استيفاء شروط مُحددة. على وجه الخصوص، يجب أن يكون عدد
  العينات "كبيرًا بما يكفي"، أو أن نماذج L1 ستعمل بشكل عشوائي،
  حيث يعتمد "كبير بما يكفي" على عدد المعاملات
  غير الصفرية، ولوغاريتم عدد الميزات، وكمية
  الضوضاء، وأصغر قيمة مُطلقة للمعاملات غير الصفرية، و
  هيكل مصفوفة التصميم X. بالإضافة إلى ذلك، يجب أن تُظهر مصفوفة التصميم
  خصائص مُحددة، مثل عدم كونها مُرتبطة للغاية.

  لا توجد قاعدة عامة لاختيار معلمة alpha لاسترداد
  المعاملات غير الصفرية. يمكن ضبطها عن طريق التحقق المتبادل
  (:class:`~sklearn.linear_model.LassoCV` أو
  :class:`~sklearn.linear_model.LassoLarsCV`)، على الرغم من أن هذا قد يؤدي إلى
  نماذج مُعاقبة بشكل ناقص: تضمين عدد صغير من المتغيرات غير ذات الصلة
  ليس ضارًا بنتيجة التنبؤ. BIC
  (:class:`~sklearn.linear_model.LassoLarsIC`) يميل، على العكس من ذلك، إلى تعيين
  قيم عالية لـ alpha.

  .. rubric:: المراجع

  Richard G. Baraniuk "الاستشعار المضغوط", IEEE Signal
  Processing Magazine [120] يوليو 2007
  http://users.isr.ist.utl.pt/~aguiar/CS_notes.pdf


اختيار الميزات القائم على الشجرة
----------------------------

يمكن استخدام المُقدِّرات القائمة على الشجرة (انظر وحدة :mod:`sklearn.tree` و
غابة الأشجار في وحدة :mod:`sklearn.ensemble`) لحساب
أهمية الميزات القائمة على النجاسة، والتي بدورها يمكن استخدامها لتجاهل
الميزات غير ذات الصلة (عند اقترانها بـ مُحوِّل التعريف
:class:`~feature_selection.SelectFromModel`)::

  >>> from sklearn.ensemble import ExtraTreesClassifier
  >>> from sklearn.datasets import load_iris
  >>> from sklearn.feature_selection import SelectFromModel
  >>> X, y = load_iris(return_X_y=True)
  >>> X.shape
  (150, 4)
  >>> clf = ExtraTreesClassifier(n_estimators=50)
  >>> clf = clf.fit(X, y)
  >>> clf.feature_importances_  # doctest: +SKIP
  array([ 0.04...,  0.05...,  0.4...,  0.4...])
  >>> model = SelectFromModel(clf, prefit=True)
  >>> X_new = model.transform(X)
  >>> X_new.shape               # doctest: +SKIP
  (150, 2)

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances.py`: مثال على
  البيانات الاصطناعية يُظهر استرداد الميزات ذات المعنى
  الفعلية.

* :ref:`sphx_glr_auto_examples_ensemble_plot_forest_importances_faces.py`: مثال
  على بيانات التعرف على الوجه.

.. _sequential_feature_selection:

اختيار الميزات المتسلسل
============================

يتوفر اختيار الميزات المتسلسل [sfs]_ (SFS) في
مُحوِّل :class:`~sklearn.feature_selection.SequentialFeatureSelector`.
يمكن أن يكون SFS إما للأمام أو للخلف:

Forward-SFS هو إجراء جشع يجد بشكل متكرر أفضل ميزة جديدة
لإضافتها إلى مجموعة الميزات المحددة. بشكل ملموس، نبدأ في البداية
بصفر ميزات ونجد الميزة الواحدة التي تُعظِّم درجة مُتحققة بشكل متبادل
عندما يتم تدريب مُقدِّر على هذه الميزة الفردية. بمجرد تحديد هذه الميزة
الأولى، نُكرر الإجراء عن طريق إضافة ميزة جديدة إلى مجموعة
الميزات المحددة. يتوقف الإجراء عندما يتم الوصول إلى العدد المطلوب من الميزات
المحددة، كما هو مُحدد بواسطة معلمة `n_features_to_select`.

يتبع Backward-SFS نفس الفكرة ولكنه يعمل في الاتجاه المُعاكس:
بدلاً من البدء بدون ميزات وإضافة الميزات بشكل جشع، نبدأ
بـ *جميع* الميزات و *نزيل* الميزات بشكل جشع من المجموعة.
تتحكم معلمة `direction` فيما إذا كان يتم استخدام SFS للأمام أو للخلف.

.. dropdown:: تفاصيل حول اختيار الميزات المتسلسل

  بشكل عام، لا يُعطي الاختيار الأمامي والخلفي نتائج مُكافئة.
  أيضًا، قد يكون أحدهما أسرع بكثير من الآخر اعتمادًا على العدد المطلوب
  من الميزات المحددة: إذا كان لدينا 10 ميزات ونطلب 7 ميزات محددة،
  فسيحتاج الاختيار الأمامي إلى إجراء 7 تكرارات بينما الاختيار الخلفي
  سيحتاج فقط إلى إجراء 3.

  يختلف SFS عن :class:`~sklearn.feature_selection.RFE` و
  :class:`~sklearn.feature_selection.SelectFromModel` من حيث أنه لا
  يتطلب أن يُظهر النموذج الأساسي سمة `coef_` أو `feature_importances_`.
  ومع ذلك، قد يكون أبطأ مع الأخذ في الاعتبار أنه يجب تقييم المزيد من النماذج،
  مُقارنةً بالطرق الأخرى. على سبيل المثال، في الاختيار
  الخلفي، يتطلب التكرار من `m` ميزات إلى `m - 1` ميزات باستخدام التحقق
  المتبادل k-fold ملاءمة `m * k` نماذج، بينما
  :class:`~sklearn.feature_selection.RFE` سيتطلب ملاءمة واحدة فقط، و
  :class:`~sklearn.feature_selection.SelectFromModel` يُجري دائمًا ملاءمة واحدة فقط
  ولا يتطلب أي تكرارات.

  .. rubric:: المراجع

  .. [sfs] Ferri et al, `دراسة مقارنة لتقنيات
      اختيار الميزات واسع النطاق
      <https://citeseerx.ist.psu.edu/doc_view/pid/5fedabbb3957bbb442802e012d829ee0629a01b6>`_.


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_feature_selection_plot_select_from_model_diabetes.py`

اختيار الميزات كجزء من خط أنابيب
=======================================

عادةً ما يُستخدم اختيار الميزات كخطوة مُعالجة مُسبقة قبل القيام
بالتعلم الفعلي. الطريقة المُوصى بها للقيام بذلك في scikit-learn
هي استخدام :class:`~pipeline.Pipeline`::

  clf = Pipeline([
    ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
    ('classification', RandomForestClassifier())
  ])
  clf.fit(X, y)

في هذا المقتطف، نستخدم :class:`~svm.LinearSVC`
مقترنًا بـ :class:`~feature_selection.SelectFromModel`
لتقييم أهمية الميزات وتحديد أهم الميزات.
ثم، يتم تدريب :class:`~ensemble.RandomForestClassifier` على
الإخراج المُحوَّل، أي باستخدام الميزات ذات الصلة فقط. يمكنك تنفيذ
عمليات مُشابهة مع أساليب اختيار الميزات الأخرى وأيضًا
المُصنِّفات التي تُوفر طريقة لتقييم أهمية الميزات بالطبع.
راجع أمثلة :class:`~pipeline.Pipeline` لمزيد من التفاصيل.


