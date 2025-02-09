
.. _gaussian_process:

==================
العمليات الغاوسية
==================

.. currentmodule:: sklearn.gaussian_process

**العمليات الغاوسية (GP)** هي أسلوب تعلم خاضع للإشراف غير بارامتري
يُستخدم لحل مشاكل *الانحدار* و *التصنيف الاحتمالي*.

مزايا العمليات الغاوسية هي:

- يتداخل التنبؤ مع الملاحظات (على الأقل بالنسبة للنوى
  المنتظمة).

- التنبؤ احتمالي (غاوسي) بحيث يمكن للمرء حساب
  فترات ثقة تجريبية واتخاذ القرار بناءً على تلك الفترات إذا كان ينبغي
  إعادة ملاءمة (ملاءمة على الإنترنت، ملاءمة تكيفية) التنبؤ في بعض
  مناطق الاهتمام.

- مُتعددة الاستخدامات: يمكن تحديد :ref:`نوى
  مُختلفة <gp_kernels>`. يتم توفير نوى شائعة، ولكن
  من الممكن أيضًا تحديد نوى مُخصصة.

تشمل عيوب العمليات الغاوسية:

- تطبيقنا ليس متفرقًا، أي أنها تستخدم معلومات العينات / الميزات
  الكاملة لإجراء التنبؤ.

- يفقدون الكفاءة في المساحات عالية الأبعاد - أي عندما يتجاوز عدد
  الميزات بضع عشرات.


.. _gpr:

انحدار العملية الغاوسية (GPR)
=================================

.. currentmodule:: sklearn.gaussian_process

:class:`GaussianProcessRegressor` يُطبق العمليات الغاوسية (GP) لـ
أغراض الانحدار. لهذا، يجب تحديد المُسبق لـ GP. سيجمع GP هذا المُسبق
ودالة الاحتمالية بناءً على عينات التدريب.
يسمح بإعطاء نهج احتمالي للتنبؤ عن طريق إعطاء المتوسط و
الانحراف المعياري كمخرجات عند التنبؤ.

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_noisy_targets_002.png
   :target: ../auto_examples/gaussian_process/plot_gpr_noisy_targets.html
   :align: center

يُفترض أن يكون متوسط المُسبق ثابتًا وصفرًا (لـ `normalize_y=False`) أو
متوسط بيانات التدريب (لـ `normalize_y=True`). يتم تحديد تغاير
المُسبق عن طريق تمرير كائن :ref:`نواة <gp_kernels>`. يتم تحسين المعلمات
الفائقة للنواة عند ملاءمة :class:`GaussianProcessRegressor`
عن طريق تعظيم الاحتمالية الهامشية اللوغاريتمية (LML) بناءً على
`optimizer` الذي تم تمريره. نظرًا لأن LML قد يحتوي على عدة نقاط مثلى محلية،
يمكن بدء المُحسِّن بشكل متكرر عن طريق تحديد `n_restarts_optimizer`. يتم
إجراء التشغيل الأول دائمًا بدءًا من قيم المعلمات الفائقة الأولية للنواة؛
يتم إجراء عمليات التشغيل اللاحقة من قيم المعلمات الفائقة التي تم اختيارها
عشوائيًا من نطاق القيم المسموح بها. إذا كانت المعلمات الفائقة الأولية
يجب أن تظل ثابتة، فيمكن تمرير `None` كـ مُحسِّن.

يمكن تحديد مستوى الضوضاء في الأهداف عن طريق تمريره عبر المعلمة
`alpha`، إما عالميًا كقيمة عددية أو لكل نقطة بيانات. لاحظ أن مستوى
الضوضاء المُعتدل يمكن أن يكون مفيدًا أيضًا للتعامل مع عدم الاستقرار العددي أثناء
الملاءمة لأنه يتم تطبيقه بشكل فعال على أنه تنظيم Tikhonov، أي عن طريق
إضافته إلى قطري مصفوفة النواة. بديل لتحديد
مستوى الضوضاء صراحةً هو تضمين مكون
:class:`~sklearn.gaussian_process.kernels.WhiteKernel` في
النواة، والذي يمكنه تقدير مستوى الضوضاء العالمي من البيانات (انظر المثال
أدناه). يُظهر الشكل أدناه تأثير الهدف الصاخب الذي يتم التعامل معه عن طريق تعيين
المعلمة `alpha`.

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_noisy_targets_003.png
   :target: ../auto_examples/gaussian_process/plot_gpr_noisy_targets.html
   :align: center

يعتمد التطبيق على الخوارزمية 2.1 من [RW2006]_. بالإضافة إلى
واجهة برمجة التطبيقات لمُقدِّرات scikit-learn القياسية، :class:`GaussianProcessRegressor`:

* يسمح بالتنبؤ بدون ملاءمة مُسبقة (بناءً على مُسبق GP)

* يُوفر أسلوبًا إضافيًا ``sample_y(X)``، والذي يُقيِّم العينات
  المرسومة من GPR (مُسبق أو لاحق) عند مدخلات مُعطاة

* يُظهِر أسلوب ``log_marginal_likelihood(theta)``، والذي يمكن استخدامه
  خارجيًا لأساليب أخرى لاختيار المعلمات الفائقة، على سبيل المثال، عبر
  Markov chain Monte Carlo.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_gaussian_process_plot_gpr_noisy_targets.py`
* :ref:`sphx_glr_auto_examples_gaussian_process_plot_gpr_noisy.py`
* :ref:`sphx_glr_auto_examples_gaussian_process_plot_compare_gpr_krr.py`
* :ref:`sphx_glr_auto_examples_gaussian_process_plot_gpr_co2.py`

.. _gpc:

تصنيف العملية الغاوسية (GPC)
=====================================

.. currentmodule:: sklearn.gaussian_process

:class:`GaussianProcessClassifier` يُطبق العمليات الغاوسية (GP) لـ
أغراض التصنيف، وبشكل أكثر تحديدًا للتصنيف الاحتمالي،
حيث تأخذ تنبؤات الاختبار شكل احتمالات الفئة.
يضع GaussianProcessClassifier مُسبق GP على دالة كامنة :math:`f`،
والتي يتم ضغطها بعد ذلك من خلال دالة ربط للحصول على
التصنيف الاحتمالي. الدالة الكامنة :math:`f` هي ما يسمى دالة إزعاج،
قيمها غير مُلاحظة وليست ذات صلة بحد ذاتها.
الغرض منها هو السماح بصياغة مُلائمة للنموذج، و :math:`f`
يتم إزالتها (دمجها) أثناء التنبؤ. GaussianProcessClassifier
يُطبق دالة ربط لوجستية، والتي لا يمكن حساب التكامل لها
تحليليًا ولكن يتم تقريبها بسهولة في الحالة الثنائية.

على عكس إعداد الانحدار، فإن التوزيع اللاحق للدالة الكامنة
:math:`f` ليس غاوسيًا حتى بالنسبة لمُسبق GP لأن الاحتمالية الغاوسية
غير مُناسبة لتصنيفات الفئات المنفصلة. بدلاً من ذلك، يتم استخدام احتمالية
غير غاوسية تقابل دالة الربط اللوجستية (logit).
يقوم GaussianProcessClassifier بتقريب التوزيع اللاحق غير الغاوسي باستخدام
توزيع غاوسي بناءً على تقريب لابلاس. يمكن العثور على مزيد من التفاصيل في
الفصل 3 من [RW2006]_.

يُفترض أن يكون متوسط مُسبق GP صفرًا. يتم تحديد
التغاير المُسبق عن طريق تمرير كائن :ref:`نواة <gp_kernels>`. يتم تحسين المعلمات
الفائقة للنواة أثناء ملاءمة GaussianProcessRegressor عن طريق تعظيم
الاحتمالية الهامشية اللوغاريتمية (LML) بناءً على ``optimizer`` الذي تم تمريره. نظرًا
لأن LML قد يحتوي على عدة نقاط مثلى محلية،
يمكن بدء المُحسِّن بشكل متكرر عن طريق تحديد ``n_restarts_optimizer``.
يتم إجراء التشغيل الأول دائمًا بدءًا من قيم المعلمات الفائقة الأولية
للنواة؛ يتم إجراء عمليات التشغيل اللاحقة من قيم المعلمات الفائقة
التي تم اختيارها عشوائيًا من نطاق القيم المسموح بها.
إذا كانت المعلمات الفائقة الأولية يجب أن تظل ثابتة، فيمكن تمرير `None` كـ
مُحسِّن.

يدعم :class:`GaussianProcessClassifier` التصنيف متعدد الفئات
عن طريق إجراء تدريب وتنبؤ قائم على واحد مقابل الباقي أو واحد مقابل واحد.
في واحد مقابل الباقي، يتم ملاءمة مُصنف عملية غاوسية ثنائية واحدة
لكل فئة، والتي يتم تدريبها لفصل هذه الفئة عن الباقي.
في "one_vs_one"، يتم ملاءمة مُصنف عملية غاوسية ثنائية واحدة لكل زوج
من الفئات، والتي يتم تدريبها لفصل هاتين الفئتين. يتم دمج تنبؤات
هذه المُتنبئات الثنائية في تنبؤات متعددة الفئات. انظر القسم الخاص بـ
:ref:`التصنيف متعدد الفئات <multiclass>` لمزيد من التفاصيل.

في حالة تصنيف العملية الغاوسية، قد يكون "one_vs_one"
أقل تكلفة من الناحية الحسابية لأنه يتعين عليه حل العديد من المشاكل التي تتضمن فقط
مجموعة فرعية من مجموعة التدريب بأكملها بدلاً من مشاكل أقل على مجموعة البيانات
بأكملها. نظرًا لأن تصنيف العملية الغاوسية يتناسب تكعيبيًا مع حجم
مجموعة البيانات، فقد يكون هذا أسرع بكثير. ومع ذلك، لاحظ أن
"one_vs_one" لا يدعم التنبؤ بتقديرات الاحتمالية ولكن فقط التنبؤات
العادية. علاوة على ذلك، لاحظ أن :class:`GaussianProcessClassifier` لا
يُطبق (حتى الآن) تقريب لابلاس متعدد الفئات حقيقيًا داخليًا، ولكن
كما نوقش أعلاه، يعتمد على حل العديد من مهام التصنيف الثنائية
داخليًا، والتي يتم دمجها باستخدام واحد مقابل الباقي أو واحد مقابل واحد.

أمثلة GPC
============

التنبؤات الاحتمالية مع GPC
----------------------------------

يوضح هذا المثال الاحتمال المتوقع لـ GPC لنواة RBF
مع اختيارات مختلفة للمعلمات الفائقة. يُظهر الشكل الأول
الاحتمال المتوقع لـ GPC مع معلمات فائقة تم اختيارها عشوائيًا ومع
المعلمات الفائقة المقابلة لأقصى احتمالية هامشية لوغاريتمية (LML).

بينما تتمتع المعلمات الفائقة التي تم اختيارها عن طريق تحسين LML بـ LML أكبر
بشكل ملحوظ، فإنها تؤدي بشكل أسوأ قليلاً وفقًا لخسارة السجل على بيانات الاختبار.
يُظهر الشكل أن هذا يرجع إلى أنها تُظهر تغيرًا حادًا في احتمالات
الفئة عند حدود الفئة (وهو أمر جيد) ولكن لديها احتمالات متوقعة
قريبة من 0.5 بعيدًا عن حدود الفئة (وهو أمر سيئ).
يحدث هذا التأثير غير المرغوب فيه بسبب تقريب لابلاس المستخدم
داخليًا بواسطة GPC.

يُظهر الشكل الثاني الاحتمالية الهامشية اللوغاريتمية لاختيارات مختلفة
من المعلمات الفائقة للنواة، مع تسليط الضوء على الخيارين لـ
المعلمات الفائقة المستخدمة في الشكل الأول بنقاط سوداء.

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc.html
   :align: center

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_002.png
   :target: ../auto_examples/gaussian_process/plot_gpc.html
   :align: center


توضيح GPC على مجموعة بيانات XOR
--------------------------------------

.. currentmodule:: sklearn.gaussian_process.kernels

يوضح هذا المثال GPC على بيانات XOR. تتم مقارنة نواة ثابتة ومتجانسة
(:class:`RBF`) ونواة غير ثابتة (:class:`DotProduct`). على
هذه المجموعة المعينة من البيانات، تحصل نواة :class:`DotProduct` على
نتائج أفضل بكثير لأن حدود الفئة خطية وتتزامن مع
محاور الإحداثيات. ومع ذلك، في الممارسة العملية، غالبًا ما تحصل النوى الثابتة مثل :class:`RBF`
على نتائج أفضل.

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_xor_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc_xor.html
   :align: center

.. currentmodule:: sklearn.gaussian_process


تصنيف العملية الغاوسية (GPC) على مجموعة بيانات iris
-----------------------------------------------------

يوضح هذا المثال الاحتمال المتوقع لـ GPC لنواة RBF متجانسة
وغير متجانسة على إصدار ثنائي الأبعاد لمجموعة بيانات iris.
هذا يوضح قابلية تطبيق GPC على التصنيف غير الثنائي.
تحصل نواة RBF غير المتجانسة على احتمالية هامشية لوغاريتمية أعلى قليلاً
عن طريق تعيين مقاييس طول مختلفة لبعدي الميزة.

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpc_iris_001.png
   :target: ../auto_examples/gaussian_process/plot_gpc_iris.html
   :align: center


.. _gp_kernels:

نوى العمليات الغاوسية
==============================
.. currentmodule:: sklearn.gaussian_process.kernels

النوى (تُسمى أيضًا "دوال التغاير" في سياق GPs) هي مُكوِّن
أساسي لـ GPs التي تُحدد شكل المُسبق واللاحق لـ GP.
إنها تُرمِّز الافتراضات على الدالة التي يتم تعلمها عن طريق تحديد "تشابه"
نقطتي بيانات مُجتمعتين مع افتراض أن نقاط البيانات المُتشابهة يجب
أن يكون لها قيم مستهدفة مُتشابهة. يمكن تمييز فئتين من النوى:
تعتمد النوى الثابتة فقط على مسافة نقطتي بيانات وليس على قيمها
المُطلقة :math:`k(x_i, x_j)= k(d(x_i, x_j))` وبالتالي فهي ثابتة لـ
الترجمات في فضاء الإدخال، بينما النوى غير الثابتة
تعتمد أيضًا على القيم المحددة لنقاط البيانات. يمكن تقسيم النوى الثابتة
إلى نوى متجانسة وغير متجانسة، حيث تكون النوى المتجانسة
أيضًا ثابتة للدوران في فضاء الإدخال. لمزيد من التفاصيل، نُشير إلى
الفصل 4 من [RW2006]_. للحصول على إرشادات حول كيفية الجمع بين النوى
المختلفة بشكل أفضل، نُشير إلى [Duv2014]_.

.. dropdown:: واجهة برمجة تطبيقات نواة العملية الغاوسية

   الاستخدام الرئيسي لـ :class:`Kernel` هو حساب تغاير GP بين
   نقاط البيانات. لهذا، يمكن استدعاء أسلوب ``__call__`` للنواة. هذا
   الأسلوب يمكن استخدامه إما لحساب "التغاير التلقائي" لجميع أزواج
   نقاط البيانات في مصفوفة ثنائية الأبعاد X، أو "التغاير المتبادل" لجميع مجموعات
   نقاط البيانات لمصفوفة ثنائية الأبعاد X مع نقاط البيانات في مصفوفة ثنائية الأبعاد Y.
   الهوية التالية صحيحة لجميع النوى k (باستثناء :class:`WhiteKernel`):
   ``k(X) == K(X, Y=X)``

   إذا تم استخدام قطري التغاير التلقائي فقط، فيمكن استدعاء أسلوب ``diag()``
   لنواة، وهو أكثر كفاءة من الناحية الحسابية من الاستدعاء المُكافئ لـ
   ``__call__``: ``np.diag(k(X, X)) == k.diag(X)``

   يتم تحديد معلمات النوى بواسطة متجه :math:`\theta` من المعلمات الفائقة. هذه
   المعلمات الفائقة يمكنها على سبيل المثال التحكم في مقاييس الطول أو دورية
   النواة (انظر أدناه). تدعم جميع النوى حساب التدرجات التحليلية
   لتغاير النواة التلقائي فيما يتعلق بـ :math:`log(\theta)` عبر التعيين
   ``eval_gradient=True`` في أسلوب ``__call__``.
   أي، يتم إرجاع مصفوفة ``(len(X), len(X), len(theta))`` حيث الإدخال
   ``[i, j, l]`` يحتوي على :math:`\frac{\partial k_\theta(x_i, x_j)}{\partial log(\theta_l)}`.
   يستخدم هذا التدرج بواسطة العملية الغاوسية (كل من المُنحدِر والمُصنف)
   في حساب تدرج الاحتمالية الهامشية اللوغاريتمية، والذي بدوره يُستخدم
   لتحديد قيمة :math:`\theta`، التي تُعظِّم الاحتمالية الهامشية اللوغاريتمية،
   عبر الصعود التدرجي. لكل معلمة فائقة، القيمة الأولية و
   الحدود يجب تحديدها عند إنشاء مثيل للنواة.
   يمكن الحصول على القيمة الحالية لـ :math:`\theta` وتعيينها عبر الخاصية
   ``theta`` لكائن النواة. علاوة على ذلك، يمكن الوصول إلى حدود المعلمات
   الفائقة بواسطة خاصية ``bounds`` للنواة. لاحظ أن كلا الخاصيتين
   (theta و bounds) تُعيدان قيمًا مُحوَّلة لوغاريتميًا للقيم المستخدمة داخليًا
   نظرًا لأنها عادةً ما تكون أكثر ملاءمة للتحسين القائم على التدرج.
   يتم تخزين مواصفات كل معلمة فائقة على شكل مثيل لـ
   :class:`Hyperparameter` في النواة المعنية. لاحظ أن النواة التي تستخدم
   معلمة فائقة باسم "x" يجب أن تحتوي على السمتين self.x و self.x_bounds.

   الفئة الأساسية المُجردة لجميع النوى هي :class:`Kernel`. تُطبق Kernel واجهة
   مُشابهة لـ :class:`~sklearn.base.BaseEstimator`، وتُوفر
   أساليب ``get_params()`` و ``set_params()`` و ``clone()``. يسمح هذا
   بتعيين قيم النواة أيضًا عبر مُقدِّرات التعريف مثل
   :class:`~sklearn.pipeline.Pipeline` أو
   :class:`~sklearn.model_selection.GridSearchCV`. لاحظ أنه نظرًا للبنية
   المُتداخلة للنوى (عن طريق تطبيق عوامل التشغيل للنواة، انظر أدناه)، فإن أسماء
   معلمات النواة قد تُصبح مُعقدة نسبيًا. بشكل عام، بالنسبة لعامل تشغيل
   النواة الثنائي، تتم إضافة بادئة ``k1__`` لمعلمات المعامل الأيسر و
   ``k2__`` لمعلمات المعامل الأيمن. أسلوب راحة إضافي
   هو ``clone_with_theta(theta)``، الذي يُعيد نسخة مُستنسخة من النواة
   ولكن مع تعيين المعلمات الفائقة إلى ``theta``. مثال توضيحي:

      >>> from sklearn.gaussian_process.kernels import ConstantKernel, RBF
      >>> kernel = ConstantKernel(constant_value=1.0, constant_value_bounds=(0.0, 10.0)) * RBF(length_scale=0.5, length_scale_bounds=(0.0, 10.0)) + RBF(length_scale=2.0, length_scale_bounds=(0.0, 10.0))
      >>> for hyperparameter in kernel.hyperparameters: print(hyperparameter)
      Hyperparameter(name='k1__k1__constant_value', value_type='numeric', bounds=array([[ 0., 10.]]), n_elements=1, fixed=False)
      Hyperparameter(name='k1__k2__length_scale', value_type='numeric', bounds=array([[ 0., 10.]]), n_elements=1, fixed=False)
      Hyperparameter(name='k2__length_scale', value_type='numeric', bounds=array([[ 0., 10.]]), n_elements=1, fixed=False)
      >>> params = kernel.get_params()
      >>> for key in sorted(params): print("%s : %s" % (key, params[key]))
      k1 : 1**2 * RBF(length_scale=0.5)
      k1__k1 : 1**2
      k1__k1__constant_value : 1.0
      k1__k1__constant_value_bounds : (0.0, 10.0)
      k1__k2 : RBF(length_scale=0.5)
      k1__k2__length_scale : 0.5
      k1__k2__length_scale_bounds : (0.0, 10.0)
      k2 : RBF(length_scale=2)
      k2__length_scale : 2.0
      k2__length_scale_bounds : (0.0, 10.0)
      >>> print(kernel.theta)  # ملاحظة: مُحوَّلة لوغاريتميًا
      [ 0.         -0.69314718  0.69314718]
      >>> print(kernel.bounds)  # ملاحظة: مُحوَّلة لوغاريتميًا
      [[      -inf 2.30258509]
      [      -inf 2.30258509]
      [      -inf 2.30258509]]

   جميع نوى العمليات الغاوسية قابلة للتشغيل البيني مع :mod:`sklearn.metrics.pairwise`
   والعكس صحيح: يمكن تمرير مثيلات الفئات الفرعية لـ :class:`Kernel` كـ
   ``metric`` إلى ``pairwise_kernels`` من :mod:`sklearn.metrics.pairwise`. علاوة على ذلك،
   يمكن استخدام دوال النواة من pairwise كنوى GP باستخدام فئة
   التغليف :class:`PairwiseKernel`. التحذير الوحيد هو أن تدرج
   المعلمات الفائقة ليس تحليليًا ولكنه رقمي وجميع تلك النوى تدعم
   المسافات المتجانسة فقط. تعتبر المعلمة ``gamma``
   معلمة فائقة ويمكن تحسينها. يتم تعيين معلمات النواة الأخرى
   مباشرةً عند التهيئة وتظل ثابتة.

النوى الأساسية
-------------
يمكن استخدام نواة :class:`ConstantKernel` كجزء من نواة :class:`Product`
حيث تُغيّر مقياس حجم العامل الآخر (النواة) أو كجزء
من نواة :class:`Sum`، حيث تُعدِّل متوسط العملية الغاوسية.
يعتمد على معلمة :math:`constant\_value`. يتم تعريفه على النحو التالي:

.. math::
   k(x_i, x_j) = constant\_value \;\forall\; x_1, x_2

حالة الاستخدام الرئيسية لنواة :class:`WhiteKernel` هي كجزء من
نواة مجموع حيث تُفسر مكون الضوضاء للإشارة. ضبط
معلمتها :math:`noise\_level` يقابل تقدير مستوى الضوضاء.
يتم تعريفه على النحو التالي:

.. math::
    k(x_i, x_j) = noise\_level \text{ if } x_i == x_j \text{ else } 0


عوامل تشغيل النواة
----------------
يأخذ عوامل تشغيل النواة نواة أساسية واحدة أو اثنتين ويجمعهما في نواة
جديدة. تأخذ نواة :class:`Sum` نواتين :math:`k_1` و :math:`k_2`
وتجمعهما عبر :math:`k_{sum}(X, Y) = k_1(X, Y) + k_2(X, Y)`.
تأخذ نواة :class:`Product` نواتين :math:`k_1` و :math:`k_2`
وتجمعهما عبر :math:`k_{product}(X, Y) = k_1(X, Y) * k_2(X, Y)`.
تأخذ نواة :class:`Exponentiation` نواة أساسية واحدة ومعلمة عددية
:math:`p` وتجمعهما عبر
:math:`k_{exp}(X, Y) = k(X, Y)^p`.
لاحظ أنه تم تجاوز الأساليب السحرية ``__add__`` و ``__mul___`` و ``__pow__``
على كائنات Kernel، لذلك يمكن للمرء استخدام على سبيل المثال ``RBF() + RBF()`` كـ
اختصار لـ ``Sum(RBF(), RBF())``.

دالة أساس نصف القطر (RBF)
----------------------------------
نواة :class:`RBF` هي نواة ثابتة. تُعرف أيضًا باسم نواة "الأس
التربيعي". يتم تحديد معلماتها بواسطة معلمة مقياس الطول :math:`l>0`، والتي
يمكن أن تكون إما عددية (متغير متجانس للنواة) أو متجه بنفس
عدد أبعاد المدخلات :math:`x` (متغير غير متجانس للنواة).
يتم إعطاء النواة بواسطة:

.. math::
   k(x_i, x_j) = \text{exp}\left(- \frac{d(x_i, x_j)^2}{2l^2} \right)

حيث :math:`d(\cdot, \cdot)` هي مسافة إقليدية.
هذه النواة قابلة للاشتقاق بلا حدود، مما يعني أن GPs بهذه
النواة كدالة تغاير لها مُشتقات مربعة متوسطة لجميع الرتب، وبالتالي
فهي سلسة جدًا. يظهر المُسبق واللاحق لـ GP الناتج عن نواة RBF في
الشكل التالي:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_001.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center


نواة Matérn
-------------
نواة :class:`Matern` هي نواة ثابتة وتعميم لـ
نواة :class:`RBF`. لديها معلمة إضافية :math:`\nu` التي تتحكم في
سلاسة الدالة الناتجة. يتم تحديد معلماتها بواسطة معلمة مقياس الطول :math:`l>0`، والتي
يمكن أن تكون إما عددية (متغير متجانس للنواة) أو متجه بنفس
عدد أبعاد المدخلات :math:`x` (متغير غير متجانس للنواة).

.. dropdown:: التطبيق الرياضي لنواة Matérn

   يتم إعطاء النواة بواسطة:

   .. math::

      k(x_i, x_j) = \frac{1}{\Gamma(\nu)2^{\nu-1}}\Bigg(\frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg)^\nu K_\nu\Bigg(\frac{\sqrt{2\nu}}{l} d(x_i , x_j )\Bigg),

   حيث :math:`d(\cdot,\cdot)` هي مسافة إقليدية، :math:`K_\nu(\cdot)` هي دالة Bessel
   مُعدلة و :math:`\Gamma(\cdot)` هي دالة جاما.
   عندما :math:`\nu\rightarrow\infty`، تتقارب نواة Matérn مع نواة RBF.
   عندما :math:`\nu = 1/2`، تُصبح نواة Matérn مطابقة لنواة
   الأس المُطلق، أي،

   .. math::
      k(x_i, x_j) = \exp \Bigg(- \frac{1}{l} d(x_i , x_j ) \Bigg) \quad \quad \nu= \tfrac{1}{2}

   على وجه الخصوص، :math:`\nu = 3/2`:

   .. math::
      k(x_i, x_j) =  \Bigg(1 + \frac{\sqrt{3}}{l} d(x_i , x_j )\Bigg) \exp \Bigg(-\frac{\sqrt{3}}{l} d(x_i , x_j ) \Bigg) \quad \quad \nu= \tfrac{3}{2}

   و :math:`\nu = 5/2`:

   .. math::
      k(x_i, x_j) = \Bigg(1 + \frac{\sqrt{5}}{l} d(x_i , x_j ) +\frac{5}{3l} d(x_i , x_j )^2 \Bigg) \exp \Bigg(-\frac{\sqrt{5}}{l} d(x_i , x_j ) \Bigg) \quad \quad \nu= \tfrac{5}{2}

   هي اختيارات شائعة لتعلم الدوال التي لا يُمكن اشتقاقها بلا حدود
   (كما هو مُفترض بواسطة نواة RBF) ولكن على الأقل مرة واحدة (:math:`\nu =
   3/2`) أو مرتين قابلة للاشتقاق (:math:`\nu = 5/2`).

   مرونة التحكم في سلاسة الدالة التي تم تعلمها عبر :math:`\nu`
   تسمح بالتكيف مع خصائص العلاقة الوظيفية الأساسية الحقيقية.

يظهر المُسبق واللاحق لـ GP الناتج عن نواة Matérn في
الشكل التالي:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_005.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

راجع [RW2006]_, pp84 لمزيد من التفاصيل حول
المتغيرات المختلفة لنواة Matérn.

نواة تربيعية عقلانية
-------------------------

يمكن اعتبار نواة :class:`RationalQuadratic` خليط مقياس (مجموع لا نهائي)
لنوى :class:`RBF` ذات مقاييس طول مميزة مختلفة. يتم تحديد
معلماتها بواسطة معلمة مقياس الطول :math:`l>0` ومعلمة خليط المقياس :math:`\alpha>0`
يتم دعم المتغير المتجانس فقط حيث :math:`l` هو عدد قياسي في الوقت الحالي.
يتم إعطاء النواة بواسطة:

.. math::
   k(x_i, x_j) = \left(1 + \frac{d(x_i, x_j)^2}{2\alpha l^2}\right)^{-\alpha}

يظهر المُسبق واللاحق لـ GP الناتج عن نواة :class:`RationalQuadratic` في
الشكل التالي:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_002.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

نواة Exp-Sine-Squared
-----------------------

تسمح نواة :class:`ExpSineSquared` بنمذجة الدوال الدورية.
يتم تحديد معلماتها بواسطة معلمة مقياس الطول :math:`l>0` ومعلمة دورية
:math:`p>0`. يتم دعم المتغير المتجانس فقط حيث :math:`l` هو عدد قياسي في الوقت الحالي.
يتم إعطاء النواة بواسطة:

.. math::
   k(x_i, x_j) = \text{exp}\left(- \frac{ 2\sin^2(\pi d(x_i, x_j) / p) }{ l^ 2} \right)

يظهر المُسبق واللاحق لـ GP الناتج عن نواة ExpSineSquared في
الشكل التالي:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_003.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

نواة Dot-Product
------------------

نواة :class:`DotProduct` غير ثابتة ويمكن الحصول عليها من الانحدار الخطي
عن طريق وضع مُسبقات :math:`N(0, 1)` على معاملات :math:`x_d (d = 1, . . . , D)`
ومُسبق :math:`N(0, \sigma_0^2)` على التحيز. نواة :class:`DotProduct` ثابتة
لدوران الإحداثيات حول الأصل، ولكن ليس للترجمات.
يتم تحديد معلماتها بواسطة معلمة :math:`\sigma_0^2`. لـ :math:`\sigma_0^2 = 0`، تُسمى النواة
النواة الخطية المتجانسة، وإلا فهي غير متجانسة. يتم إعطاء النواة بواسطة

.. math::
   k(x_i, x_j) = \sigma_0 ^ 2 + x_i \cdot x_j

عادةً ما يتم دمج نواة :class:`DotProduct` مع الأس. مثال بأس 2
موضح في الشكل التالي:

.. figure:: ../auto_examples/gaussian_process/images/sphx_glr_plot_gpr_prior_posterior_004.png
   :target: ../auto_examples/gaussian_process/plot_gpr_prior_posterior.html
   :align: center

المراجع
----------

.. [RW2006] `Carl E. Rasmussen and Christopher K.I. Williams,
   "العمليات الغاوسية للتعلم الآلي",
   MIT Press 2006 <https://www.gaussianprocess.org/gpml/chapters/RW.pdf>`_

.. [Duv2014] `David Duvenaud, "كتاب طبخ النواة: نصائح حول دوال التغاير", 2014
   <https://www.cs.toronto.edu/~duvenaud/cookbook/>`_

.. currentmodule:: sklearn.gaussian_process


