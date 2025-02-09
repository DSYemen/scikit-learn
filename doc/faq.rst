.. raw:: html

  <style>
    /* h3 headings on this page are the questions; make them rubric-like */
    h3 {
      font-size: 1rem;
      font-weight: bold;
      padding-bottom: 0.2rem;
      margin: 2rem 0 1.15rem 0;
      border-bottom: 1px solid var(--pst-color-border);
    }

    /* Increase top margin for first question in each section */
    h2 + section > h3 {
      margin-top: 2.5rem;
    }

    /* Make the headerlinks a bit more visible */
    h3 > a.headerlink {
      font-size: 0.9rem;
    }

    /* Remove the backlink decoration on the titles */
    h2 > a.toc-backref,
    h3 > a.toc-backref {
      text-decoration: none;
    }
  </style>

.. _faq:

==========================
الأسئلة المتكررة
==========================

.. currentmodule:: sklearn

نحاول هنا تقديم بعض الإجابات على الأسئلة التي تظهر بانتظام في القائمة البريدية.

.. contents:: جدول المحتويات
  :local:
  :depth: 2


حول المشروع
-----------------

ما اسم المشروع (يخطئ الكثير من الناس في ذلك)؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
scikit-learn، وليس scikit أو SciKit ولا sci-kit learn.
أيضًا ليس scikits.learn أو scikits-learn، والتي تم استخدامها سابقًا.

كيف تنطق اسم المشروع؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
sy-kit learn. sci اختصار لـ science!

لماذا scikit؟
^^^^^^^^^^^
هناك العديد من scikits، وهي عبارة عن صناديق أدوات علمية مبنية حول SciPy.
بصرف النظر عن scikit-learn، هناك صندوق أدوات شهير آخر هو `scikit-image <https://scikit-image.org/>`_.

هل تدعمون PyPy؟
^^^^^^^^^^^^^^^^^^^^

نظرًا لموارد الصيانة المحدودة وقلة عدد المستخدمين، فإن استخدام
scikit-learn مع `PyPy <https://pypy.org/>`_ (تنفيذ بديل لبايثون مع مترجم
مضمن في الوقت المناسب) غير مدعوم رسميًا.


كيف يمكنني الحصول على إذن لاستخدام الصور في scikit-learn لعملي؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يمكن استخدام الصور الموجودة في `مستودع scikit-learn
<https://github.com/scikit-learn/scikit-learn>`_ والصور التي تم إنشاؤها ضمن
`وثائق scikit-learn <https://scikit-learn.org/stable/index.html>`_
عبر `رخصة BSD 3-Clause
<https://github.com/scikit-learn/scikit-learn?tab=BSD-3-Clause-1-ov-file>`_ لعملك.
نشجع ونقدر بشدة الاستشهادات بـ scikit-learn. انظر
:ref:`الاستشهاد بـ scikit-learn <citing-scikit-learn>`.


قرارات التنفيذ
------------------------

لماذا لا يوجد دعم للتعلم العميق أو التعلم المعزز؟ هل سيكون هناك مثل هذا الدعم في المستقبل؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يتطلب كل من التعلم العميق والتعلم المعزز مفردات غنية لتعريف بنية،
مع حاجة التعلم العميق بالإضافة إلى ذلك إلى وحدات معالجة الرسومات للحوسبة
الفعالة. ومع ذلك، لا يتناسب أي منهما مع قيود تصميم scikit-learn. نتيجة
لذلك، يقع التعلم العميق والتعلم المعزز حاليًا خارج نطاق ما تسعى
scikit-learn إلى تحقيقه.

يمكنك العثور على مزيد من المعلومات حول إضافة دعم وحدة معالجة الرسومات في
`هل ستضيفون دعم وحدة معالجة الرسومات؟`_.

لاحظ أن scikit-learn تنفذ حاليًا شبكة عصبية متعددة الطبقات بسيطة
في :mod:`sklearn.neural_network`. سنقبل فقط إصلاحات الأخطاء لهذه الوحدة.
إذا كنت ترغب في تنفيذ نماذج تعلم عميق أكثر تعقيدًا، فيرجى الرجوع إلى
أطر عمل التعلم العميق الشائعة مثل
`tensorflow <https://www.tensorflow.org/>`_،
`keras <https://keras.io/>`_،
و `pytorch <https://pytorch.org/>`_.

.. _adding_graphical_models:

هل ستضيفون نماذج رسومية أو تنبؤ بالتسلسل إلى scikit-learn؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

ليس في المستقبل المنظور.
تحاول scikit-learn توفير واجهة برمجة تطبيقات موحدة للمهام الأساسية في التعلم
الآلي، مع خطوط أنابيب وخوارزميات وصفية مثل بحث الشبكة لربط كل شيء معًا.
تختلف المفاهيم المطلوبة وواجهات برمجة التطبيقات والخوارزميات والخبرات
المطلوبة للتعلم المنظم عما تقدمه scikit-learn. إذا بدأنا في
إجراء تعلم منظم تعسفي، فسنحتاج إلى إعادة تصميم الحزمة بأكملها، ومن المرجح أن
ينهار المشروع تحت ثقله.

هناك مشروعان لهما واجهة برمجة تطبيقات مشابهة لـ scikit-learn يقومان بالتنبؤ المنظم:

* `pystruct <https://pystruct.github.io/>`_ يتعامل مع التعلم المنظم العام (يركز على
  SSVMs على هياكل رسوم بيانية عشوائية مع استدلال تقريبي؛ يعرّف فكرة العينة
  كحالة لهيكل الرسم البياني).

* `seqlearn <https://larsmans.github.io/seqlearn/>`_ يتعامل مع التسلسلات فقط
  (يركز على الاستدلال الدقيق؛ لديه HMMs، ولكن في الغالب من أجل
  الاكتمال؛ يعامل متجه الميزات كعينة ويستخدم ترميز الإزاحة للتبعيات
  بين متجهات الميزات).

لماذا أزلتم HMMs من scikit-learn؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
انظر :ref:`adding_graphical_models`.


هل ستضيفون دعم وحدة معالجة الرسومات؟
^^^^^^^^^^^^^^^^^^^^^^^^^

ستؤدي إضافة دعم وحدة معالجة الرسومات افتراضيًا إلى إدخال تبعيات برمجية
خاصة بالأجهزة الثقيلة، وستحتاج الخوارزميات الحالية إلى إعادة تنفيذها. سيجعل
هذا الأمر أكثر صعوبة لكل من المستخدم العادي لتثبيت scikit-learn وللمطورين
لصيانة الكود.

ومع ذلك، منذ عام 2023، فإن :ref:`قائمة محدودة ولكن متنامية من مقدرات scikit-learn
<array_api_supported>` يمكنها بالفعل العمل على وحدات معالجة الرسومات إذا تم توفير بيانات
الإدخال كمصفوفة PyTorch أو CuPy وإذا تم تكوين scikit-learn لقبول
مثل هذه المدخلات كما هو موضح في :ref:`array_api`. يسمح دعم Array API
لـ scikit-learn بالعمل على وحدات معالجة الرسومات دون إدخال تبعيات
برمجية ثقيلة وخاصة بالأجهزة للحزمة الرئيسية.

يمكن اعتبار معظم المقدرات التي تعتمد على NumPy لعملياتها الحسابية
المكثفة لدعم Array API وبالتالي دعم وحدة معالجة الرسومات.

ومع ذلك، لا يمكن لجميع مقدرات scikit-learn العمل بكفاءة على وحدات
معالجة الرسومات عبر Array API لأسباب خوارزمية أساسية. على سبيل المثال،
نماذج تعتمد على الشجرة يتم تنفيذها حاليًا مع Cython في scikit-learn
ليست أساسًا خوارزميات تعتمد على المصفوفات. تعتمد الخوارزميات الأخرى مثل
k-means أو k-nearest neighbors على خوارزميات تعتمد على المصفوفات ولكنها
يتم تنفيذها أيضًا في Cython. يتم استخدام Cython لتشبيك عمليات المصفوفة
المتتالية يدويًا لتجنب إدخال الوصول إلى الذاكرة الذي يقتل الأداء
لمصفوفات وسيطة كبيرة: تُعرف إعادة الكتابة الخوارزمية منخفضة المستوى
هذه باسم "دمج النواة" ولا يمكن التعبير عنها عبر Array API في المستقبل
المنظور.

ستتطلب إضافة دعم وحدة معالجة الرسومات بكفاءة إلى المقدرات التي لا يمكن
تنفيذها بكفاءة باستخدام Array API تصميم واعتماد نظام توسيع أكثر مرونة
لـ scikit-learn. يتم النظر في هذا الاحتمال في مشكلة GitHub التالية (قيد المناقشة):

- https://github.com/scikit-learn/scikit-learn/issues/22438


لماذا تحتاج المتغيرات الفئوية إلى المعالجة المسبقة في scikit-learn، مقارنة بالأدوات الأخرى؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

تفترض معظم scikit-learn أن البيانات موجودة في مصفوفات NumPy أو مصفوفات
SciPy المتفرقة من نوع بيانات رقمي واحد. هذه لا تمثل المتغيرات الفئوية
صراحة في الوقت الحاضر. وبالتالي، على عكس ``data.frames`` فيR أو :class:`pandas.DataFrame`، نطلب تحويلًا صريحًا للميزات الفئوية إلى قيم رقمية، كما هو موضح في :ref:`preprocessing_categorical_features`.
راجع أيضًا :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py` للحصول على مثال للعمل مع بيانات غير متجانسة (مثل الفئوية والرقمية).

لاحظ أنه في الآونة الأخيرة، اكتسب كل من :class:`~sklearn.ensemble.HistGradientBoostingClassifier` و :class:`~sklearn.ensemble.HistGradientBoostingRegressor` دعمًا أصليًا للميزات الفئوية من خلال الخيار `categorical_features="from_dtype"`. يعتمد هذا الخيار على استنتاج أعمدة البيانات الفئوية بناءً على أنواع البيانات :class:`pandas.CategoricalDtype` و :class:`polars.datatypes.Categorical`.

هل تعمل scikit-learn محليًا مع أنواع مختلفة من إطارات البيانات؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

لدى Scikit-learn دعم محدود لـ :class:`pandas.DataFrame` و :class:`polars.DataFrame`. يمكن لمقدرات Scikit-learn قبول كلا نوعي إطار البيانات كمدخلات، ويمكن لمحولات Scikit-learn إخراج إطارات البيانات باستخدام واجهة برمجة تطبيقات `set_output`. لمزيد من التفاصيل، يرجى الرجوع إلى :ref:`sphx_glr_auto_examples_miscellaneous_plot_set_output.py`.

ومع ذلك، تعتمد الحسابات الداخلية في مقدرات Scikit-learn على العمليات العددية التي يتم إجراؤها بشكل أكثر كفاءة على هياكل البيانات المتجانسة مثل مصفوفات NumPy أو مصفوفات SciPy المتفرقة. نتيجة لذلك، ستقوم معظم مقدرات Scikit-learn داخليًا بتحويل مدخلات إطار البيانات إلى هياكل البيانات المتجانسة هذه. وبالمثل، يتم إنشاء مخرجات إطار البيانات من هياكل البيانات المتجانسة هذه.

لاحظ أيضًا أن :class:`~sklearn.compose.ColumnTransformer` يجعل من الملائم التعامل مع إطارات بيانات الباندا غير المتجانسة عن طريق تعيين مجموعات فرعية متجانسة من أعمدة إطار البيانات المحددة بالاسم أو نوع البيانات إلى محولات Scikit-learn مخصصة. لذلك، غالبًا ما يتم استخدام :class:`~sklearn.compose.ColumnTransformer` في الخطوة الأولى من خطوط أنابيب Scikit-learn عند التعامل مع إطارات البيانات غير المتجانسة (انظر :ref:`pipeline` لمزيد من التفاصيل).

راجع أيضًا :ref:`sphx_glr_auto_examples_compose_plot_column_transformer_mixed_types.py` للحصول على مثال للعمل مع بيانات غير متجانسة (مثل الفئوية والرقمية).


هل تخططون لتنفيذ تحويل للهدف ``y`` في خط أنابيب؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
يعمل التحويل حاليًا فقط للميزات ``X`` في خط أنابيب. هناك نقاش طويل الأمد حول عدم القدرة على تحويل ``y`` في خط أنابيب. تابع مشكلة GitHub :issue:`4143`. في غضون ذلك، يمكنك مراجعة :class:`~compose.TransformedTargetRegressor` و `pipegraph <https://github.com/mcasl/PipeGraph>`_ و `imbalanced-learn <https://github.com/scikit-learn-contrib/imbalanced-learn>`_. لاحظ أن scikit-learn قد حلت الحالة التي يتم فيها تطبيق تحويل قابل للعكس على ``y`` قبل التدريب وعكسه بعد التنبؤ. تنوي scikit-learn حل حالات الاستخدام التي يجب فيها تحويل ``y`` في وقت التدريب وليس في وقت الاختبار، لإعادة التشكيل واستخدامات مماثلة، كما هو الحال في `imbalanced-learn <https://github.com/scikit-learn-contrib/imbalanced-learn>`_. بشكل عام، يمكن حل حالات الاستخدام هذه باستخدام مقدر وصفية مخصص بدلاً من :class:`~pipeline.Pipeline`.

لماذا يوجد الكثير من المقدرات المختلفة للنماذج الخطية؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
عادةً، يوجد مصنف واحد ومنظم واحد لكل نوع نموذج، على سبيل المثال :class:`~ensemble.GradientBoostingClassifier` و :class:`~ensemble.GradientBoostingRegressor`. كلاهما لهما خيارات متشابهة وكلاهما لديه المعلمة `loss`، وهو أمر مفيد بشكل خاص في حالة الانحدار لأنه يتيح تقدير المتوسط ​​الشرطي بالإضافة إلى الكميات الشرطية.

بالنسبة للنماذج الخطية، هناك العديد من فئات المقدرات القريبة جدًا من بعضها البعض. دعونا نلقي نظرة على

- :class:`~linear_model.LinearRegression`، بدون عقوبة
- :class:`~linear_model.Ridge`، عقوبة L2
- :class:`~linear_model.Lasso`، عقوبة L1 (نماذج متفرقة)
- :class:`~linear_model.ElasticNet`، عقوبة L1 + L2 (نماذج أقل تفرقًا)
- :class:`~linear_model.SGDRegressor` مع `loss="squared_loss"`

**منظور المسؤول عن الصيانة:**
جميعهم يفعلون من حيث المبدأ نفس الشيء ويختلفون فقط من خلال العقوبة التي يفرضونها. ومع ذلك، فإن هذا له تأثير كبير على الطريقة التي يتم بها حل مشكلة التحسين الأساسية. في النهاية، يرقى هذا إلى استخدام طرق وحيل مختلفة من الجبر الخطي. حالة خاصة هي :class:`~linear_model.SGDRegressor` التي تضم جميع النماذج الأربعة السابقة وتختلف من خلال إجراء التحسين. أحد الآثار الجانبية الإضافية هو أن المقدرات المختلفة تفضل تخطيطات بيانات مختلفة (`X` متجاورة C أو متجاورة F، متفرقة csr أو csc). هذا التعقيد للنماذج الخطية التي تبدو بسيطة هو سبب وجود فئات مقدرات مختلفة لعقوبات مختلفة.

**منظور المستخدم:**
أولاً، التصميم الحالي مستوحى من الأدبيات العلمية حيث تم إعطاء نماذج الانحدار الخطي ذات التنظيم/العقوبة المختلفة أسماء مختلفة، على سبيل المثال *انحدار ريدج*. إن وجود فئات نماذج مختلفة بأسماء متوافقة يسهل على المستخدمين العثور على نماذج الانحدار تلك.
ثانيًا، إذا تم توحيد جميع النماذج الخطية الخمسة المذكورة أعلاه في فئة واحدة، فسيكون هناك معلمات تحتوي على الكثير من الخيارات مثل معلمة ``solver``. علاوة على ذلك، سيكون هناك الكثير من التفاعلات الحصرية بين المعلمات المختلفة. على سبيل المثال، ستعتمد الخيارات الممكنة لمعلمات ``solver`` و ``precompute`` و ``selection`` على القيم المختارة لمعلمات العقوبة ``alpha`` و ``l1_ratio``.


المساهمة
------------

كيف يمكنني المساهمة في scikit-learn؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
انظر :ref:`contributing`. قبل الرغبة في إضافة خوارزمية جديدة، والتي عادة ما تكون مهمة كبيرة وطويلة، يوصى بالبدء بـ :ref:`المشكلات المعروفة <new_contributors>`. يرجى عدم الاتصال بالمساهمين في scikit-learn مباشرةً فيما يتعلق بالمساهمة في scikit-learn.

لماذا لا يحظى طلب السحب الخاص بي بأي اهتمام؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

تستغرق عملية مراجعة scikit-learn قدرًا كبيرًا من الوقت، ويجب ألا يثبط عزيمة المساهمين عدم وجود نشاط أو مراجعة لطلب السحب الخاص بهم. نحن نهتم كثيرًا بصحة الأمور في المرة الأولى، حيث أن الصيانة والتغيير اللاحق يأتي بتكلفة عالية. نادرًا ما نصدر أي كود "تجريبي"، لذلك ستخضع جميع مساهماتنا للاستخدام العالي على الفور ويجب أن تكون بأعلى جودة ممكنة في البداية.

بخلاف ذلك، فإن scikit-learn محدودة في عرض النطاق الترددي للمراجعة؛ يعمل العديد من المراجعين ومطوري النواة على scikit-learn في وقتهم الخاص. إذا كانت مراجعة طلب السحب الخاص بك بطيئة، فمن المحتمل أن يكون ذلك بسبب انشغال المراجعين. نطلب منك تفهمك ونطلب منك عدم إغلاق طلب السحب الخاص بك أو التوقف عن عملك فقط بسبب هذا السبب.

.. _new_algorithms_inclusion_criteria:

ما هي معايير تضمين الخوارزميات الجديدة؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

نحن نأخذ في الاعتبار فقط الخوارزميات الراسخة للتضمين. قاعدة عامة هي 3 سنوات على الأقل منذ النشر، و 200+ استشهاد، والاستخدام الواسع والفائدة. سيتم أيضًا النظر في تضمين تقنية توفر تحسينًا واضحًا (على سبيل المثال، بنية بيانات محسنة أو تقنية تقريب أكثر كفاءة) على طريقة مستخدمة على نطاق واسع.

من الخوارزميات أو التقنيات التي تستوفي المعايير المذكورة أعلاه، يتم قبول فقط تلك التي تتناسب جيدًا مع واجهة برمجة تطبيقات scikit-learn الحالية، أي واجهة ``fit`` و ``predict/transform`` وعادةً ما يكون الإدخال/الإخراج عبارة عن مصفوفة numpy أو مصفوفة متفرقة.

يجب على المساهم دعم أهمية الإضافة المقترحة بأوراق بحثية و/أو تطبيقات في حزم أخرى مماثلة، وإظهار فائدتها من خلال حالات الاستخدام/التطبيقات الشائعة وتأكيد تحسينات الأداء، إن وجدت، من خلال المعايير و/أو الرسوم البيانية. من المتوقع أن تتفوق الخوارزمية المقترحة على الطرق التي تم تنفيذها بالفعل في scikit-learn على الأقل في بعض المجالات.

يكون تضمين خوارزمية جديدة لتسريع نموذج موجود أسهل إذا:

- لم تقدم معلمات تشعبية جديدة (لأنها تجعل المكتبة أكثر مقاومة للمستقبل)،
- من السهل توثيق متى يحسن المساهمة السرعة ومتى لا يحسنها بوضوح، على سبيل المثال، "عندما ``n_features >> n_samples``"،
- تُظهر المعايير بوضوح تسريعًا.

لاحظ أيضًا أن تطبيقك لا يحتاج إلى أن يكون في scikit-learn ليتم استخدامه مع أدوات scikit-learn. يمكنك تنفيذ خوارزميتك المفضلة بطريقة متوافقة مع scikit-learn، وتحميلها إلى GitHub وإخبارنا بذلك. سنكون سعداء بإدراجه ضمن :ref:`related_projects`. إذا كانت لديك بالفعل حزمة على GitHub تتبع واجهة برمجة تطبيقات scikit-learn، فقد تكون مهتمًا أيضًا بالاطلاع على `scikit-learn-contrib <https://scikit-learn-contrib.github.io>`_.

.. _selectiveness:

لماذا أنت انتقائي للغاية بشأن الخوارزميات التي تدرجها في scikit-learn؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
يأتي الكود بتكلفة صيانة، ونحن بحاجة إلى موازنة كمية الكود التي لدينا مع حجم الفريق (وأضف إلى ذلك حقيقة أن التعقيد يتناسب بشكل غير خطي مع عدد الميزات). تعتمد الحزمة على مطوري النواة الذين يستخدمون وقتهم الخاص لإصلاح الأخطاء وصيانة الكود ومراجعة المساهمات. أي خوارزمية تمت إضافتها تحتاج إلى اهتمام مستقبلي من قبل المطورين، وعند هذه النقطة قد يكون المؤلف الأصلي قد فقد الاهتمام منذ فترة طويلة. انظر أيضًا :ref:`new_algorithms_inclusion_criteria`. لقراءة رائعة حول مشكلات الصيانة طويلة المدى في برمجيات المصادر المفتوحة، انظر `الملخص التنفيذي للطرق والجسور <https://www.fordfoundation.org/media/2976/roads-and-bridges-the-unseen-labor-behind-our-digital-infrastructure.pdf#page=8>`_.


استخدام scikit-learn
------------------

ما هي أفضل طريقة للحصول على مساعدة بشأن استخدام scikit-learn؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* أسئلة التعلم الآلي العامة: استخدم `Cross Validated <https://stats.stackexchange.com/>`_ مع علامة ``[machine-learning]``.

* أسئلة استخدام scikit-learn: استخدم `Stack Overflow <https://stackoverflow.com/questions/tagged/scikit-learn>`_ مع علامات ``[scikit-learn]`` و ``[python]``. يمكنك بدلاً من ذلك استخدام `القائمة البريدية <https://mail.python.org/mailman/listinfo/scikit-learn>`_.

يرجى التأكد من تضمين مقتطف شفرة إعادة إنتاج صغير (يفضل أن يكون أقصر من 10 أسطر) يبرز مشكلتك على مجموعة بيانات لعبة (على سبيل المثال من :mod:`sklearn.datasets` أو تم إنشاؤها عشوائيًا باستخدام وظائف ``numpy.random`` مع بذرة عشوائية ثابتة). يرجى إزالة أي سطر من التعليمات البرمجية غير الضروري لإعادة إنتاج مشكلتك.

يجب أن تكون المشكلة قابلة للتكرار بمجرد نسخ مقتطف الشفرة ولصقه في غلاف بايثون مع تثبيت scikit-learn. لا تنس تضمين عبارات الاستيراد. يمكن العثور على مزيد من الإرشادات لكتابة مقتطفات تعليمات برمجية جيدة لإعادة الإنتاج على: https://stackoverflow.com/help/mcve.

إذا أثارت مشكلتك استثناءً لا تفهمه (حتى بعد البحث عنه في جوجل)، فيرجى التأكد من تضمين التتبع الكامل الذي تحصل عليه عند تشغيل البرنامج النصي لإعادة الإنتاج.

لتقارير الأخطاء أو طلبات الميزات، يرجى استخدام `متتبع المشكلات على GitHub <https://github.com/scikit-learn/scikit-learn/issues>`_.

.. warning::
  يرجى عدم إرسال بريد إلكتروني إلى أي مؤلفين مباشرة لطلب المساعدة أو الإبلاغ عن الأخطاء أو أي مشكلة أخرى تتعلق بـ scikit-learn.

كيف يمكنني حفظ أو تصدير أو نشر المقدرات للإنتاج؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

انظر :ref:`model_persistence`.


كيف يمكنني إنشاء كائن حزمة؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

تُستخدم كائنات الحزمة أحيانًا كمخرج للوظائف والأساليب. وهي توسع القواميس عن طريق تمكين الوصول إلى القيم بواسطة مفتاح، `bunch["value_key"]`، أو بواسطة سمة، `bunch.value_key`.

لا ينبغي استخدامها كمدخلات. لذلك لا تحتاج أبدًا إلى إنشاء كائن :class:`~utils.Bunch`، إلا إذا كنت تقوم بتوسيع واجهة برمجة تطبيقات scikit-learn.


كيف يمكنني تحميل مجموعات البيانات الخاصة بي بتنسيق قابل للاستخدام بواسطة scikit-learn؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

بشكل عام، تعمل scikit-learn على أي بيانات رقمية مخزنة كمصفوفات numpy أو مصفوفات scipy المتفرقة. الأنواع الأخرى القابلة للتحويل إلى مصفوفات رقمية مثل :class:`pandas.DataFrame` مقبولة أيضًا.

لمزيد من المعلومات حول تحميل ملفات البيانات الخاصة بك في هياكل البيانات القابلة للاستخدام هذه، يرجى الرجوع إلى :ref:`تحميل مجموعات البيانات الخارجية <external_datasets>`.


كيف أتعامل مع بيانات السلسلة (أو الأشجار، الرسوم البيانية...)؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

تفترض مقدرات scikit-learn أنك ستغذيها متجهات ميزات ذات قيمة حقيقية. هذا الافتراض مشفر بشكل ثابت في كل مكتبة تقريبًا. ومع ذلك، يمكنك تغذية المدخلات غير الرقمية إلى المقدرات بعدة طرق.

إذا كانت لديك مستندات نصية، فيمكنك استخدام ميزات تكرار المصطلحات؛ راجع :ref:`text_feature_extraction` لـ *متجهات النص* المضمنة. لمزيد من استخراج الميزات العامة من أي نوع من البيانات، راجع :ref:`dict_feature_extraction` و :ref:`feature_hashing`.

حالة شائعة أخرى هي عندما يكون لديك بيانات غير رقمية ومقياس مسافة (أو تشابه) مخصص على هذه البيانات. تتضمن الأمثلة سلاسل ذات مسافة تحرير (المعروفة أيضًا باسم مسافة ليفنشتاين)، على سبيل المثال، تسلسلات الحمض النووي أو الحمض النووي الريبي. يمكن ترميزها كأرقام، لكن القيام بذلك أمر مؤلم وعرضة للخطأ. يمكن العمل مع مقاييس المسافة على البيانات التعسفية بطريقتين.

أولاً، تأخذ العديد من المقدرات مصفوفات المسافة/التشابه المحسوبة مسبقًا، لذلك إذا لم تكن مجموعة البيانات كبيرة جدًا، فيمكنك حساب المسافات لجميع أزواج المدخلات. إذا كانت مجموعة البيانات كبيرة، فيمكنك استخدام متجهات الميزات بميزة واحدة فقط، وهي فهرس في بنية بيانات منفصلة، وتوفير دالة مقياس مخصصة تبحث عن البيانات الفعلية في بنية البيانات هذه. على سبيل المثال، لاستخدام :class:`~cluster.dbscan` مع مسافات ليفنشتاين::

    >>> import numpy as np
    >>> from leven import levenshtein  # doctest: +SKIP
    >>> from sklearn.cluster import dbscan
    >>> data = ["ACCTCCTAGAAG", "ACCTACTAGAAGTT", "GAATATTAGGCCGA"]
    >>> def lev_metric(x, y):
    ...     i, j = int(x[0]), int(y[0])  # extract indices
    ...     return levenshtein(data[i], data[j])
    ...
    >>> X = np.arange(len(data)).reshape(-1, 1)
    >>> X
    array([[0],
           [1],
           [2]])
    >>> # We need to specify algorithm='brute' as the default assumes
    >>> # a continuous feature space.
    >>> dbscan(X, metric=lev_metric, eps=5, min_samples=2, algorithm='brute')  # doctest: +SKIP
    (array([0, 1]), array([ 0,  0, -1]))

لاحظ أن المثال أعلاه يستخدم حزمة مسافة التحرير الخارجية `leven <https://pypi.org/project/leven/>`_. يمكن استخدام حيل مماثلة، مع بعض العناية، لنوى الأشجار، ونوى الرسوم البيانية، وما إلى ذلك.


لماذا أحصل أحيانًا على تعطل/تجميد مع ``n_jobs > 1`` تحت OSX أو Linux؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

تعتمد العديد من أدوات scikit-learn مثل :class:`~model_selection.GridSearchCV` و :class:`~model_selection.cross_val_score` داخليًا على وحدة :mod:`multiprocessing` في بايثون لموازنة التنفيذ على عدة عمليات بايثون عن طريق تمرير ``n_jobs > 1`` كوسيطة.

المشكلة هي أن :mod:`multiprocessing` في بايثون تقوم باستدعاء نظام ``fork`` دون متابعته باستدعاء نظام ``exec`` لأسباب تتعلق بالأداء. تدير العديد من المكتبات مثل (بعض إصدارات) Accelerate أو vecLib تحت OSX، (بعض إصدارات) MKL، وقت تشغيل OpenMP لـ GCC، Cuda من nvidia (وربما العديد من المكتبات الأخرى)، تجمع سلاسل العمليات الداخلي الخاص بها. عند استدعاء `fork`، تتلف حالة تجمع سلاسل العمليات في العملية الفرعية: يعتقد تجمع سلاسل العمليات أن لديه العديد من سلاسل العمليات بينما تم تفرع حالة سلسلة العمليات الرئيسية فقط. من الممكن تغيير المكتبات لجعلها تكتشف متى يحدث التفرع وإعادة تهيئة تجمع سلاسل العمليات في هذه الحالة: لقد فعلنا ذلك لـ OpenBLAS (تم دمجه في المنبع في الرئيسي منذ 0.2.10) وساهمنا بـ `تصحيح <https://gcc.gnu.org/bugzilla/show_bug.cgi?id=60035>`_ لوقت تشغيل OpenMP لـ GCC (لم تتم مراجعته بعد).

لكن في النهاية، الجاني الحقيقي هو :mod:`multiprocessing` في بايثون التي تقوم بـ ``fork`` بدون ``exec`` لتقليل الحمل الزائد لبدء واستخدام عمليات بايثون جديدة للحوسبة المتوازية. لسوء الحظ، هذا يعد انتهاكًا لمعيار POSIX، وبالتالي يرفض بعض محرري البرامج مثل Apple اعتبار عدم أمان التفرع في Accelerate و vecLib كخطأ.

في بايثون 3.4+، أصبح من الممكن الآن تكوين :mod:`multiprocessing` لاستخدام أساليب بدء ``"forkserver"`` أو ``"spawn"`` (بدلاً من ``"fork"`` الافتراضية) لإدارة تجمعات العمليات. للتغلب على هذه المشكلة عند استخدام scikit-learn، يمكنك تعيين متغير البيئة ``JOBLIB_START_METHOD`` إلى ``"forkserver"``. ومع ذلك، يجب أن يكون المستخدم على دراية بأن استخدام طريقة ``"forkserver"`` يمنع :class:`joblib.Parallel` من استدعاء دالة محددة بشكل تفاعلي في جلسة غلاف.

إذا كان لديك كود مخصص يستخدم :mod:`multiprocessing` مباشرةً بدلاً من استخدامه عبر :mod:`joblib`، فيمكنك تمكين وضع ``"forkserver"`` عالميًا لبرنامجك. أدخل التعليمات التالية في البرنامج النصي الرئيسي الخاص بك::

    import multiprocessing

    # استيرادات أخرى، كود مخصص، تحميل بيانات، تعريف نموذج...

    if __name__ == "__main__":
        multiprocessing.set_start_method("forkserver")

        # استدعاء أدوات scikit-learn مع n_jobs > 1 هنا

يمكنك العثور على المزيد من الإعدادات الافتراضية حول أساليب البدء الجديدة في `وثائق المعالجة المتعددة <https://docs.python.org/3/library/multiprocessing.html#contexts-and-start-methods>`_.

.. _faq_mkl_threading:

لماذا تستخدم وظيفتي نوى أكثر مما هو محدد بواسطة ``n_jobs``؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

هذا لأن ``n_jobs`` يتحكم فقط في عدد الوظائف للروتينات المتوازنة بواسطة :mod:`joblib`، لكن الكود المتوازي يمكن أن يأتي من مصادر أخرى:

- يمكن موازنة بعض الروتينات مع OpenMP (للكود المكتوب بلغة C أو Cython)،
- تعتمد scikit-learn كثيرًا على numpy، والتي بدورها قد تعتمد على مكتبات رقمية مثل MKL أو OpenBLAS أو BLIS والتي يمكنها توفير تطبيقات متوازية.

لمزيد من التفاصيل، يرجى الرجوع إلى :ref:`ملاحظاتنا حول التوازي <parallelism>`.


كيف يمكنني تعيين ``random_state`` لتنفيذ كامل؟
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

يرجى الرجوع إلى :ref:`randomness`.

