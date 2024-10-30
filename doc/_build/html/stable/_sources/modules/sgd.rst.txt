.. _sgd:

===========================
التحسين التدريجي العشوائي
===========================

.. currentmodule:: sklearn.linear_model

**التحسين التدريجي العشوائي (SGD)** هو نهج بسيط وفعال للغاية
لتناسب المصنفات والمرجعيات الخطية في ظل
وظائف الخسارة المقعرة مثل (الخطية) `دعم الآلات المتجهة
<https://en.wikipedia.org/wiki/Support_vector_machine>`_ و `الانحدار اللوجستي
<https://en.wikipedia.org/wiki/Logistic_regression>`_.
على الرغم من أن SGD موجود في مجتمع التعلم الآلي منذ فترة طويلة، إلا أنه تلقى قدرًا كبيرًا من الاهتمام مؤخرًا
في سياق التعلم على نطاق واسع.

تم تطبيق SGD بنجاح على مشكلات التعلم الآلي الكبيرة والنادرة التي يتم مواجهتها غالبًا في تصنيف النصوص ومعالجة اللغات الطبيعية.  نظرًا لأن البيانات نادرة، فإن المصنفات
في هذه الوحدة النمطية تتناسب بسهولة مع المشكلات التي تحتوي على أكثر من 10^5 أمثلة تدريبية وأكثر من 10^5 ميزات.

بمعنى دقيق، فإن SGD هو مجرد تقنية تحسين ولا
تقابل عائلة محددة من نماذج التعلم الآلي. إنه مجرد
*طريقة* لتدريب نموذج. غالبًا ما يكون مثيل :class:`SGDClassifier` أو
:class:`SGDRegressor` له مصنف مكافئ في
واجهة برمجة scikit-learn، ربما باستخدام تقنية تحسين مختلفة.
على سبيل المثال، يؤدي استخدام `SGDClassifier(loss='log_loss')` إلى الانحدار اللوجستي،
أي نموذج مكافئ لـ :class:`~sklearn.linear_model.LogisticRegression`
الذي يتم تجهيزه عبر SGD بدلاً من تجهيزه بواسطة أحد الحلول الأخرى
في :class:`~sklearn.linear_model.LogisticRegression`. وبالمثل،
`SGDRegressor(loss='squared_error', penalty='l2')` و
:class:`~sklearn.linear_model.Ridge` تحل نفس مشكلة التحسين، عبر
طرق مختلفة.

مزايا التحسين التدريجي العشوائي هي:

+ الكفاءة.

+ سهولة التنفيذ (الكثير من الفرص لتحسين الكود).

عيوب التحسين التدريجي العشوائي تشمل:

+ يتطلب SGD عددًا من المعلمات الفائقة مثل معلمة التنظيم
وعدد التكرارات.

+ SGD حساس لقياس الميزات.

.. warning::

  تأكد من تبديل (خلط) بيانات التدريب الخاصة بك قبل تجهيز النموذج
  أو استخدام ``shuffle=True`` لخلطها بعد كل تكرار (يستخدم افتراضيًا).
  أيضًا، من الناحية المثالية، يجب توحيد الميزات باستخدام e.g.
  `make_pipeline(StandardScaler(), SGDClassifier())` (راجع :ref:`Pipelines
  <combining_estimators>`).

التصنيف
==============


تنفذ الفئة :class:`SGDClassifier` روتين تعلم التحسين التدريجي العشوائي العادي الذي يدعم وظائف الخسارة والعقوبات المختلفة للتصنيف. فيما يلي حدود قرار :class:`SGDClassifier` المدرب باستخدام خسارة الهامش، المكافئ لمصنف SVM الخطي.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_separating_hyperplane_001.png
   :target: ../auto_examples/linear_model/plot_sgd_separating_hyperplane.html
   :align: center
   :scale: 75

بعد التجهيز، يمكن استخدام النموذج بعد ذلك للتنبؤ بالقيم الجديدة::

    >>> clf.predict([[2., 2.]])
    array([1])

يتناسب SGD مع نموذج خطي لبيانات التدريب. تحتوي السمة ``coef_`` على
معلمات النموذج::

    >>> clf.coef_
    array([[9.9..., 9.9...]])

تحتوي السمة ``intercept_`` على الانترسبت::

    >>> clf.intercept_
    array([-9.9...])

يتم التحكم فيما إذا كان يجب على النموذج استخدام انترسبت، أي فرضية متحيزة،
بمعلمة ``fit_intercept``.

المسافة الموقعة إلى الفرضية (المحسوبة كناتج الضرب النقطي
بين المعاملات والعينة المدخلة، بالإضافة إلى الانترسبت) يتم إعطاؤها بواسطة
:meth:`SGDClassifier.decision_function`::

    >>> clf.decision_function([[2., 2.]])
    array([29.6...])

يمكن تعيين وظيفة الخسارة الملموسة عبر معلمة ``loss``
. يدعم :class:`SGDClassifier` وظائف الخسارة التالية:

* ``loss="hinge"``: (soft-margin) SVM الخطي،
* ``loss="modified_huber"``: خسارة الهامش الملساء،
* ``loss="log_loss"``: الانحدار اللوجستي،
* وجميع وظائف الانحدار أدناه. في هذه الحالة، يتم ترميز الهدف على أنه -1
  أو 1، وتتم معاملة المشكلة على أنها مشكلة انحدار. الفئة المتوقعة
  تقابل إشارة الهدف المتوقع.

يرجى الرجوع إلى القسم الرياضي أدناه
<sgd_mathematical_formulation>` للحصول على الصيغ.
وظيفتا الخسارة الأوليتان كسولتان، فهما لا تحدثان سوى تحديث نموذج
المعلمات إذا انتهكت إحدى الأمثلة قيد الهامش، مما يجعل
التدريب فعالاً للغاية وقد يؤدي إلى نماذج أكثر ندرة (أي مع المزيد من المعاملات الصفرية)، حتى عند استخدام عقوبة L2.

باستخدام ``loss="log_loss"`` أو ``loss="modified_huber"``، يتم تمكين
طريقة ``predict_proba``، والتي تعطي ناقلًا من تقديرات الاحتمالية
:math:`P(y|x)` لكل عينة :math:`x`::

    >>> clf = SGDClassifier(loss="log_loss", max_iter=5).fit(X, y)
    >>> clf.predict_proba([[1., 1.]]) # doctest: +SKIP
    array([[0.00..., 0.99...]])

يمكن تعيين العقوبة الملموسة عبر معلمة ``penalty``
. يدعم SGD العقوبات التالية:

* ``penalty="l2"``: معيار L2 على ``coef_``.
* ``penalty="l1"``: معيار L1 على ``coef_``.
* ``penalty="elasticnet"``: مزيج محدب من L2 وL1؛
  ``(1 - l1_ratio) * L2 + l1_ratio * L1``.

الإعداد الافتراضي هو ``penalty="l2"``. تؤدي عقوبة L1 إلى حلول نادرة،
مما يؤدي إلى دفع معظم المعاملات إلى الصفر. يحل Elastic Net [#5]_ بعض أوجه القصور في عقوبة L1 في وجود سمات شديدة الارتباط. يتحكم المعامل ``l1_ratio`` في المزيج المحدب
من L1 وL2.

يدعم :class:`SGDClassifier` التصنيف متعدد الفئات من خلال الجمع
بين العديد من المصنفات الثنائية في مخطط "واحد مقابل الكل" (OVA). لكل
من الفئات :math:`K`، يتم تعلم مصنف ثنائي يميز
بين تلك الفئات وجميع الفئات الأخرى :math:`K-1`. في وقت الاختبار، نقوم بحساب
درجة الثقة (أي المسافة الموقعة إلى الفرضية) لكل
مصنف واختيار الفئة ذات الثقة الأعلى. يوضح الشكل
أدناه نهج OVA على مجموعة بيانات زهرة النرجس.  تمثل الخطوط المتقطعة
المصنفات الثلاثة OVA؛ الألوان الخلفية تظهر
سطح القرار الناجم عن المصنفات الثلاثة.

.. figure:: ../auto_examples/linear_model/images/sphx_glr_plot_sgd_iris_001.png
   :target: ../auto_examples/linear_model/plot_sgd_iris.html
   :align: center
   :scale: 75

في حالة التصنيف متعدد الفئات، يكون ``coef_`` مصفوفة ثنائية الأبعاد
ذات الشكل (n_classes، n_features) و ``intercept_`` هو
مصفوفة أحادية البعد ذات الشكل (n_classes,). يحتوي الصف i-th من ``coef_`` على
متجه الوزن لمصنف OVA للفئة i-th؛ يتم ترتيب الفئات
تصاعديًا (راجع سمة ``classes_``).
لاحظ أنه، من حيث المبدأ، نظرًا لأنها تسمح بإنشاء نموذج احتمالي،
``loss="log_loss"`` و ``loss="modified_huber"`` أكثر ملاءمة
للتصنيف "واحد مقابل الكل".

يدعم :class:`SGDClassifier` كل من الفئات المرجحة والعينات المرجحة عبر معلمات التجهيز ``class_weight`` و ``sample_weight``. راجع
الأمثلة أدناه وdocstring لـ :meth:`SGDClassifier.fit` للحصول
على مزيد من المعلومات.

يدعم :class:`SGDClassifier` متوسط SGD [#4]_. يمكن تمكين المتوسط من خلال تعيين `average=True`. يقوم ASGD بأداء نفس التحديثات مثل
SGD العادي (راجع :ref:`sgd_mathematical_formulation`)، ولكن بدلاً من استخدام
القيمة الأخيرة لمعاملات كسمة ``coef_`` (أي قيم
آخر تحديث)، يتم تعيين ``coef_`` بدلاً من ذلك إلى **متوسط** قيمة
المعاملات عبر جميع التحديثات. نفس الشيء ينطبق على السمة ``intercept_``. عند استخدام ASGD، يمكن أن يكون معدل التعلم أكبر وحتى ثابتًا،
مما يؤدي على بعض مجموعات البيانات إلى تسريع وقت التدريب.

بالنسبة للتصنيف مع خسارة لوجستية، هناك متغير آخر من SGD مع استراتيجية متوسطة
متاحة مع خوارزمية Stochastic Average Gradient (SAG)،
متاحة كمصنف في :class:`LogisticRegression`.

.. rubric:: أمثلة

- :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_separating_hyperplane.py`
- :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_iris.py`
- :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_weighted_samples.py`
- :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_comparison.py`
- :ref:`sphx_glr_auto_examples_svm_plot_separating_hyperplane_unbalanced.py`
  (راجع الملاحظة في المثال)

الانحدار
==========

تنفذ الفئة :class:`SGDRegressor` روتين تعلم التحسين التدريجي العشوائي العادي الذي يدعم وظائف الخسارة والعقوبات المختلفة لتناسب نماذج الانحدار الخطية. :class:`SGDRegressor` مناسب
لمشكلات الانحدار التي تحتوي على عدد كبير من
أمثلة التدريب (> 10.000)، بالنسبة للمشكلات الأخرى نوصي بـ :class:`Ridge`،
:class:`Lasso`، أو :class:`ElasticNet`.

يمكن تعيين وظيفة الخسارة الملموسة عبر معلمة ``loss``
. يدعم :class:`SGDRegressor` وظائف الخسارة التالية:

* ``loss="squared_error"``: أقل مربعات عادية،
* ``loss="huber"``: خسارة هابر للانحدار المقاوم،
* ``loss="epsilon_insensitive"``: دعم الآلات المتجهة الخطية.

يرجى الرجوع إلى القسم الرياضي أدناه
<sgd_mathematical_formulation>` للحصول على الصيغ.
يمكن استخدام وظيفتي هابر وإبسيلون غير الحساسة للانحدار المقاوم. يجب تحديد عرض المنطقة غير الحساسة
عبر معلمة ``epsilon``. تعتمد هذه المعلمة على
مقياس المتغيرات المستهدفة.

تحدد معلمة `penalty` التنظيم المراد استخدامه (راجع
الوصف أعلاه في قسم التصنيف).

يدعم :class:`SGDRegressor` أيضًا متوسط SGD [#4]_ (هنا مرة أخرى، راجع
الوصف أعلاه في قسم التصنيف).

بالنسبة للانحدار مع خسارة مربعة وعقوبة L2، هناك متغير آخر من
SGD مع استراتيجية متوسطة متاحة مع خوارزمية Stochastic Average
Gradient (SAG)، متاحة كمصنف في :class:`Ridge`.

.. _sgd_online_one_class_svm:

Online One-Class SVM
====================

تنفذ الفئة :class:`sklearn.linear_model.SGDOneClassSVM` إصدارًا خطيًا عبر الإنترنت
من SVM فئة واحدة باستخدام التحسين التدريجي العشوائي.
عند الجمع مع تقنيات تقريب النواة،
يمكن استخدام :class:`sklearn.linear_model.SGDOneClassSVM` لتقريب حل
SVM فئة واحدة المطبقة على نواة، والتي تم تنفيذها في
:class:`sklearn.svm.OneClassSVM`، مع تعقيد خطي
في عدد العينات. لاحظ أن التعقيد لSVM فئة واحدة المطبقة على نواة هو على
الأفضل تربيعي في عدد العينات.
:class:`sklearn.linear_model.SGDOneClassSVM` مناسب
لذلك لمجموعات البيانات التي تحتوي على عدد كبير من
أمثلة التدريب (> 10,000) والتي يمكن أن يكون متغير SGD
أسرع بعدة أوامر من حيث الحجم.

.. dropdown:: التفاصيل الرياضية

  يعتمد تنفيذه على تنفيذ التحسين التدريجي العشوائي. في الواقع، مشكلة التحسين الأصلية لSVM فئة واحدة هي

  .. math::

    \begin{aligned}
    \min_{w, \rho, \xi} & \quad \frac{1}{2}\Vert w \Vert^2 - \rho + \frac{1}{\nu n} \sum_{i=1}^n \xi_i \\
    \text{s.t.} & \quad \langle w, x_i \rangle \geq \rho - \xi_i \quad 1 \leq i \leq n \\
    & \quad \xi_i \geq 0 \quad 1 \leq i \leq n
    \end{aligned}

  حيث :math:`\nu \in (0, 1]` هو معلمة المستخدم المحددة التي تتحكم في
  نسبة الشذوذ ونسبة المتجهات الداعمة. التخلص من
  المتغيرات السائبة :math:`\xi_i` هذه المشكلة مكافئة لـ

  .. math::

    \min_{w, \rho} \frac{1}{2}\Vert w \Vert^2 - \rho + \frac{1}{\nu n} \sum_{i=1}^n \max(0, \rho - \langle w, x_i \rangle) \, .

  الضرب في الثابت :math:`\nu` وإدخال الانترسبت
  :math:`b = 1 - \rho` نحصل على مشكلة التحسين المكافئة التالية

  .. math::

    \min_{w, b} \frac{\nu}{2}\Vert w \Vert^2 + b\nu + \frac{1}{n} \sum_{i=1}^n \max(0, 1 - (\langle w, x_i \rangle + b)) \, .

  هذا مشابه لمشكلات التحسين التي تمت دراستها في القسم
  :ref:`sgd_mathematical_formulation` مع :math:`y_i = 1, 1 \leq i \leq n` و
  :math:`\alpha = \nu/2`، :math:`L` كونها وظيفة الخسارة الهامشية و :math:`R`
  كونها معيار L2. نحتاج فقط إلى إضافة المصطلح :math:`b\nu` في
  حلقة التحسين.

كما هو الحال في :class:`SGDClassifier` و :class:`SGDRegressor`، يدعم :class:`SGDOneClassSVM` متوسط SGD. يمكن تمكين المتوسط من خلال تعيين ``average=True``.

.. rubric:: أمثلة

- :ref:`sphx_glr_auto_examples_linear_model_plot_sgdocsvm_vs_ocsvm.py`

التحسين التدريجي العشوائي للبيانات النادرة
===========================================

.. note:: ينتج التنفيذ النادر نتائج مختلفة قليلاً
  من التنفيذ الكثيف، بسبب معدل التعلم المصغر للانترسبت. راجع :ref:`implementation_details`.

هناك دعم مدمج للبيانات النادرة المقدمة في أي مصفوفة بتنسيق
مدعوم بواسطة `scipy.sparse
<https://docs.scipy.org/doc/scipy/reference/sparse.html>`_. للحصول على أقصى
كفاءة، ومع ذلك، استخدم تنسيق المصفوفة CSR
كما هو محدد في `scipy.sparse.csr_matrix
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.csr_matrix.html>`_.

.. rubric:: أمثلة

- :ref:`sphx_glr_auto_examples_text_plot_document_classification_20newsgroups.py`

التعقيد
==========

الميزة الرئيسية لـ SGD هي كفاءتها، والتي تكون أساسية
خطي في عدد أمثلة التدريب. إذا كانت X مصفوفة من الحجم (n, p)
يبلغ تكلفة التدريب :math:`O(k n \bar p)`، حيث k هو العدد
من التكرارات (عصور) و :math:`\bar p` هو متوسط عدد
السمات غير الصفرية لكل عينة.

ومع ذلك، تظهر النتائج النظرية الحديثة أن وقت التشغيل للحصول على بعض
دقة التحسين المطلوبة لا تزيد مع زيادة حجم مجموعة التدريب.

معيار التوقف
==================

توفر الفئتان :class:`SGDClassifier` و :class:`SGDRegressor` معيارين
لإيقاف الخوارزمية عندما يتم الوصول إلى مستوى معين من التقارب:

* مع ``early_stopping=True``، يتم تقسيم بيانات الإدخال إلى مجموعة تدريب
  ومجموعة التحقق. يتم بعد ذلك تجهيز النموذج على مجموعة التدريب، ويتم
  تحديد معيار التوقف بناءً على درجة التنبؤ (باستخدام طريقة `score`
  ) المحسوبة على مجموعة التحقق. يمكن تحديد حجم مجموعة التحقق
  عبر معلمة ``validation_fraction``.
* مع ``early_stopping=False``، يتم تجهيز النموذج على بيانات الإدخال بأكملها
  ويتم تحديد معيار التوقف بناءً على دالة الهدف المحسوبة على
  بيانات التدريب.

في كلتا الحالتين، يتم تقييم المعيار مرة واحدة لكل عصر، وتتوقف الخوارزمية
عندما لا يتحسن المعيار ``n_iter_no_change`` مرات متتالية. يتم تقييم التحسن
مع التسامح المطلق ``tol``، وتتوقف الخوارزمية في أي حال بعد عدد أقصى من التكرارات ``max_iter``.

راجع :ref:`sphx_glr_auto_examples_linear_model_plot_sgd_early_stopping.py` لمثال على آثار التوقف المبكر.

نصائح للاستخدام العملي
=====================

* التحسين التدريجي العشوائي حساس لقياس الميزات، لذا
  من المستحسن جدًا قياس بياناتك. على سبيل المثال، قم بتصنيف كل
  سمة على ناقل الإدخال X إلى [0,1] أو [-1,+1]، أو قم بتوحيد
  باستخدام e.g.
  `make_pipeline(StandardScaler(), SGDClassifier())` (راجع :ref:`Pipelines
  <combining_estimators>`).

* العثور على معلمة تنظيم معقولة :math:`\alpha`
  يتم ذلك بشكل أفضل باستخدام البحث التلقائي عن المعلمات، مثل
  :class:`~sklearn.model_selection.GridSearchCV` أو
  :class:`~sklearn.model_selection.RandomizedSearchCV`، عادة في
  النطاق ``10.0**-np.arange(1,7)``.

* وجدنا تجريبيًا أن SGD يتقارب بعد ملاحظة
  تقريبًا 10^6 أمثلة التدريب. وبالتالي، فإن تخمين أولي
  لعدد التكرارات هو ``max_iter = np.ceil(10**6 / n)`،
  حيث ``n`` هو حجم مجموعة التدريب.

* إذا كنت تطبق SGD على ميزات مستخرجة باستخدام PCA وجدنا أنه
  من الحكمة غالبًا قياس قيم الميزة بثابت `c`
  بحيث يكون متوسط L2 norm لبيانات التدريب يساوي واحد.

* وجدنا أن متوسط SGD يعمل بشكل أفضل مع عدد أكبر من الميزات
  ومعدل أعلى من eta0.

.. rubric:: المراجع

.. [#1] `"Stochastic Gradient Descent"
  <https://leon.bottou.org/projects/sgd>`_ L. Bottou - Website, 2010.

.. [#2] :doi:`"Pegasos: Primal estimated sub-gradient solver for svm"
  <10.1145/1273496.1273598>`
  S. Shalev-Shwartz، Y. Singer، N. Srebro - في وقائع ICML '07.

.. [#3] `"Stochastic gradient descent training for l1-regularized
  log-linear models with cumulative penalty"
  <https://www.aclweb.org/anthology/P/P09/P09-1054.pdf>`_
  Y. Tsuruoka، J. Tsujii، S. Ananiadou - في وقائع AFNLP/ACL'09.

.. [#4] :arxiv:`"Towards Optimal One Pass Large Scale Learning with
  Averaged Stochastic Gradient Descent"
  <1107.2490v2>`. Xu، Wei (2011)

.. [#5] :doi:`"Regularization and variable selection via the elastic net"
  <10.1111/j.1467-9868.2005.00503.x>`
  H. Zou، T. Hastie - Journal of the Royal Statistical Society Series B،
  67 (2)، 301-320.

.. [#6] :doi:`"Solving large scale linear prediction problems using stochastic
  gradient descent algorithms" <10.1145/1015330.1015332>`
  T. Zhang - في وقائع ICML '04.