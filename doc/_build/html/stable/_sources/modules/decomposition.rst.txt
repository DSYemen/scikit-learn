.. _decompositions:


=====================================================
تحليل الإشارات إلى مكونات (مشاكل تحليل المصفوفات)
=====================================================

.. currentmodule:: sklearn.decomposition


.. _PCA:


تحليل المكونات الرئيسية (PCA)
==================================

PCA التحليل الأساسي الدقيق والتفسير الاحتمالي
-----------------------------------------------

يُستخدم PCA لتحليل مجموعة بيانات متعددة المتغيرات إلى مجموعة من
المكونات المتعامدة المتتالية التي تُفسر أقصى قدر من التباين. في
scikit-learn، يتم تنفيذ :class:`PCA` ككائن *مُحوِّل*
يتعلم :math:`n` مكونات في أسلوبه ``fit``، ويمكن استخدامه على بيانات جديدة لـ
إسقاطها على هذه المكونات.

يُتمركز PCA ولكنه لا يُغيّر مقياس بيانات الإدخال لكل ميزة قبل
تطبيق SVD. تُتيح المعلمة الاختيارية ``whiten=True``
إمكانية إسقاط البيانات على الفضاء الفردي مع تغيير مقياس كل
مكون إلى تباين الوحدة. غالبًا ما يكون هذا مفيدًا إذا كانت النماذج في اتجاه مجرى النهر
تضع افتراضات قوية على تجانس الخواص للإشارة: هذا هو الحال على سبيل المثال
بالنسبة لآلات متجه الدعم مع نواة RBF وخوارزمية التجميع K-Means.

فيما يلي مثال على مجموعة بيانات iris، التي تتكون من 4
ميزات، مُسقطة على البُعدين اللذين يُفسران معظم التباين:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_lda_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_lda.html
    :align: center
    :scale: 75%


يُوفر كائن :class:`PCA` أيضًا
تفسيرًا احتماليًا لـ PCA يمكن أن يُعطي احتمالية
للبيانات بناءً على مقدار التباين الذي تُفسره. على هذا النحو، فإنه يُطبق
أسلوب :term:`score` يمكن استخدامه في التحقق المتبادل:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_001.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_pca_iris.py`
* :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_lda.py`
* :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`


.. _IncrementalPCA:

Incremental PCA
---------------

كائن :class:`PCA` مفيد جدًا، لكنه يحتوي على قيود مُعينة لـ
مجموعات البيانات الكبيرة. القيد الأكبر هو أن :class:`PCA` يدعم فقط
المعالجة الدفعية، مما يعني أن جميع البيانات التي سيتم معالجتها يجب أن تتناسب مع الذاكرة الرئيسية.
يستخدم كائن :class:`IncrementalPCA` شكلًا مختلفًا من
المعالجة ويسمح بحسابات جزئية تتطابق تقريبًا
مع نتائج :class:`PCA` أثناء معالجة البيانات بطريقة
دفعية صغيرة. :class:`IncrementalPCA` يُتيح تنفيذ
تحليل المكونات الرئيسية خارج النواة إما عن طريق:

* استخدام أسلوبه ``partial_fit`` على أجزاء من البيانات التي تم جلبها بالتسلسل
  من القرص الصلب المحلي أو قاعدة بيانات الشبكة.

* استدعاء أسلوبه fit على ملف مُخصص للذاكرة باستخدام
  ``numpy.memmap``.

:class:`IncrementalPCA` يُخزِّن فقط تقديرات مكون وتباينات الضوضاء،
من أجل تحديث ``explain_variance_ratio_`` بشكل تدريجي. هذا هو السبب
في أن استخدام الذاكرة يعتمد على عدد العينات لكل دفعة، بدلاً من
عدد العينات التي سيتم معالجتها في مجموعة البيانات.

كما هو الحال في :class:`PCA`، :class:`IncrementalPCA` يُتمركز ولكنه لا يُغيّر مقياس
بيانات الإدخال لكل ميزة قبل تطبيق SVD.

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_incremental_pca_001.png
    :target: ../auto_examples/decomposition/plot_incremental_pca.html
    :align: center
    :scale: 75%

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_incremental_pca_002.png
    :target: ../auto_examples/decomposition/plot_incremental_pca.html
    :align: center
    :scale: 75%


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_incremental_pca.py`


.. _RandomizedPCA:

PCA باستخدام SVD العشوائي
---------------------------

غالبًا ما يكون من المثير للاهتمام إسقاط البيانات على
فضاء ذي أبعاد أقل يحافظ على معظم التباين، عن طريق إسقاط المتجه الفردي
للمكونات المرتبطة بقيم فردية أقل.

على سبيل المثال، إذا كنا نعمل مع صور ذات مستوى رمادي 64x64 بكسل
للتعرف على الوجه،
فإن أبعاد البيانات هي 4096 ويكون تدريب
آلة متجه دعم RBF على مثل هذه البيانات الواسعة بطيئًا. علاوة على ذلك، نحن نعلم
أن أبعاد البيانات الجوهرية أقل بكثير من 4096 نظرًا لأن جميع
صور الوجوه البشرية تبدو متشابهة إلى حد ما.
تقع العينات على مُشعب ذي أبعاد
أقل بكثير (على سبيل المثال حوالي 200). يمكن استخدام خوارزمية PCA
لتحويل البيانات خطيًا مع تقليل الأبعاد
والحفاظ على معظم التباين المُفسَّر في نفس الوقت.

فئة :class:`PCA` المُستخدمة مع المعلمة الاختيارية
``svd_solver='randomized'`` مفيدة جدًا في هذه الحالة: نظرًا لأننا سنقوم بإسقاط معظم المتجهات الفردية، فمن الأكثر فعالية بكثير تقييد
الحساب إلى تقدير تقريبي للمتجهات الفردية التي سنحتفظ بها لـ
أداء التحويل فعليًا.

على سبيل المثال، يُظهر ما يلي 16 صورة شخصية عينة (مُتمركزة حول
0.0) من مجموعة بيانات Olivetti. على الجانب الأيمن توجد أول 16
متجهًا فرديًا أعيد تشكيلها كصور شخصية. نظرًا لأننا نطلب فقط أفضل 16
متجهًا فرديًا لمجموعة بيانات بحجم :math:`n_{samples} = 400`
و :math:`n_{features} = 64 \times 64 = 4096`، فإن وقت الحساب
أقل من 1 ثانية:

.. |orig_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_001.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. |pca_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |orig_img| |pca_img|

إذا لاحظنا :math:`n_{\max} = \max(n_{\mathrm{samples}}, n_{\mathrm{features}})` و
:math:`n_{\min} = \min(n_{\mathrm{samples}}, n_{\mathrm{features}})`، فإن التعقيد الزمني
لـ :class:`PCA` العشوائي هو :math:`O(n_{\max}^2 \cdot n_{\mathrm{components}})`
بدلاً من :math:`O(n_{\max}^2 \cdot n_{\min})` للأسلوب الدقيق
المُطبق في :class:`PCA`.

بصمة الذاكرة لـ :class:`PCA` العشوائي تتناسب أيضًا مع
:math:`2 \cdot n_{\max} \cdot n_{\mathrm{components}}` بدلاً من :math:`n_{\max}
\cdot n_{\min}` للأسلوب الدقيق.

ملاحظة: تطبيق ``inverse_transform`` في :class:`PCA` مع
``svd_solver='randomized'`` ليس التحويل العكسي الدقيق لـ
``transform`` حتى عندما يكون ``whiten=False`` (افتراضي).


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_applications_plot_face_recognition.py`
* :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. rubric:: المراجع

* الخوارزمية 4.3 في
  :arxiv:`"إيجاد الهيكل مع العشوائية: الخوارزميات العشوائية لـ
  بناء تحليلات مصفوفة تقريبية" <0909.4061>`
  Halko, et al., 2009

* :arxiv:`"تطبيق خوارزمية عشوائية لتحليل المكونات
  الرئيسية" <1412.3510>` A. Szlam et al. 2014

.. _SparsePCA:

تحليل المكونات الرئيسية المتفرقة (SparsePCA و MiniBatchSparsePCA)
-----------------------------------------------------------------------

:class:`SparsePCA` هو متغير من PCA، بهدف استخراج
مجموعة المكونات المتفرقة التي تُعيد بناء البيانات بشكل أفضل.

Mini-batch sparse PCA (:class:`MiniBatchSparsePCA`) هو متغير من
:class:`SparsePCA` وهو أسرع ولكنه أقل دقة. يتم الوصول إلى السرعة المتزايدة
عن طريق التكرار على أجزاء صغيرة من مجموعة الميزات، لعدد مُعين
من التكرارات.


تحليل المكونات الرئيسية (:class:`PCA`) له عيب أن
المكونات التي يستخرجها هذا الأسلوب لها تعبيرات كثيفة حصريًا، أي
لها معاملات غير صفرية عند التعبير عنها كمجموعات خطية من
المتغيرات الأصلية. هذا يمكن أن يجعل التفسير صعبًا. في كثير من الحالات،
يمكن تخيل المكونات الأساسية الحقيقية بشكل أكثر طبيعية كـ
متجهات متفرقة؛ على سبيل المثال في التعرف على الوجه، قد تُعيَّن المكونات بشكل طبيعي لـ
أجزاء من الوجوه.

تُنتج المكونات الرئيسية المتفرقة تمثيلًا أكثر اقتصادًا وقابلية
للتفسير، مع التأكيد بوضوح على أي من الميزات الأصلية تُساهم
في الاختلافات بين العينات.

يوضح المثال التالي 16 مكونًا تم استخراجها باستخدام PCA المتفرق من
مجموعة بيانات وجوه Olivetti. يمكن ملاحظة كيف يُحفز مُصطلح التنظيم
العديد من الأصفار. علاوة على ذلك، تتسبب البنية الطبيعية للبيانات في أن تكون المعاملات غير
الصفرية متجاورة رأسيًا. لا يفرض النموذج هذا
رياضيًا: كل مكون هو متجه :math:`h \in \mathbf{R}^{4096}`، و
لا يوجد مفهوم للتجاور الرأسي إلا أثناء التصور الصديق
للإنسان كصور بكسل 64x64. حقيقة أن المكونات الموضحة أدناه
تظهر محلية هي تأثير البنية الكامنة في البيانات، مما يجعل
مثل هذه الأنماط المحلية تُقلل من خطأ إعادة البناء. توجد معايير مُحفزة
للتفرق تأخذ في الاعتبار التجاور وأنواع مختلفة من الهياكل؛ راجع
[Jen09]_ لمراجعة هذه الأساليب.
لمزيد من التفاصيل حول كيفية استخدام Sparse PCA، راجع قسم الأمثلة، أدناه.


.. |spca_img| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_005.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |pca_img| |spca_img|

لاحظ أن هناك العديد من الصيغ المختلفة لمشكلة Sparse PCA.
الصيغة المُطبقة هنا تستند إلى [Mrl09]_. مشكلة التحسين
التي تم حلها هي مشكلة PCA (تعلم القاموس) مع
عقوبة :math:`\ell_1` على المكونات:

.. math::
   (U^*, V^*) = \underset{U, V}{\operatorname{arg\,min\,}} & \frac{1}{2}
                ||X-UV||_{\text{Fro}}^2+\alpha||V||_{1,1} \\
                \text{subject to } & ||U_k||_2 <= 1 \text{ for all }
                0 \leq k < n_{components}

:math:`||.||_{\text{Fro}}` تعني قاعدة Frobenius و :math:`||.||_{1,1}`
تعني قاعدة المصفوفة حسب الإدخال وهي مجموع القيم المطلقة
لجميع الإدخالات في المصفوفة.
تمنع قاعدة المصفوفة :math:`||.||_{1,1}` المُحفزة للتفرق أيضًا تعلم
المكونات من الضوضاء عند توفر عدد قليل من عينات التدريب. الدرجة
من العقوبة (وبالتالي التفرق) يمكن تعديلها من خلال
المعلمة الفائقة ``alpha``. تؤدي القيم الصغيرة إلى
تحليل مُنظَّم برفق، بينما تُقلِّص القيم الأكبر العديد من المعاملات إلى الصفر.

.. note::

  بينما بروح خوارزمية على الإنترنت، فإن الفئة
  :class:`MiniBatchSparsePCA` لا تُطبق ``partial_fit`` لأن
  الخوارزمية على الإنترنت على طول اتجاه الميزات، وليس اتجاه العينات.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`

.. rubric:: المراجع

.. [Mrl09] `"تعلم القاموس على الإنترنت للترميز المتفرق"
   <https://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
   J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009
.. [Jen09] `"تحليل المكونات الرئيسية المتفرقة المُهيكلة"
   <https://www.di.ens.fr/~fbach/sspca_AISTATS2010.pdf>`_
   R. Jenatton, G. Obozinski, F. Bach, 2009


.. _kernel_PCA:

تحليل المكونات الرئيسية للنواة (kPCA)
==========================================

kPCA الدقيق
----------------

:class:`KernelPCA` هو امتداد لـ PCA يحقق تقليلًا غير خطي
للأبعاد من خلال استخدام النوى (انظر :ref:`metrics`) [Scholkopf1997]_. لديها العديد من التطبيقات بما في ذلك إزالة الضوضاء والضغط والتنبؤ
المُهيكل (تقدير تبعية النواة). :class:`KernelPCA` يدعم كلاً من
``transform`` و ``inverse_transform``.

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_kernel_pca_002.png
    :target: ../auto_examples/decomposition/plot_kernel_pca.html
    :align: center
    :scale: 75%

.. note::
    :meth:`KernelPCA.inverse_transform` يعتمد على حافة النواة لتعلم تعيين الدالة
    العينات من أساس PCA إلى مساحة الميزة الأصلية [Bakir2003]_. وبالتالي، إعادة البناء التي تم الحصول عليها باستخدام
    :meth:`KernelPCA.inverse_transform` هي تقريبية. راجع المثال
    المرتبط أدناه لمزيد من التفاصيل.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_kernel_pca.py`
* :ref:`sphx_glr_auto_examples_applications_plot_digits_denoising.py`

.. rubric:: المراجع

.. [Scholkopf1997] Schölkopf, Bernhard, Alexander Smola, and Klaus-Robert Müller.
   `"تحليل المكونات الرئيسية للنواة."
   <https://people.eecs.berkeley.edu/~wainwrig/stat241b/scholkopf_kernel.pdf>`_
   المؤتمر الدولي حول الشبكات العصبية الاصطناعية.
   Springer, Berlin, Heidelberg, 1997.

.. [Bakir2003] Bakır, Gökhan H., Jason Weston, and Bernhard Schölkopf.
   `"التعلم لإيجاد صور مُسبقة."
   <https://papers.nips.cc/paper/2003/file/ac1ad983e08ad3304a97e147f522747e-Paper.pdf>`_
   التقدم في أنظمة معالجة المعلومات العصبية 16 (2003): 449-456.

.. _kPCA_Solvers:

اختيار محلل لـ Kernel PCA
-------------------------------

بينما في :class:`PCA` يكون عدد المكونات مُحددًا بعدد
الميزات، في :class:`KernelPCA` يكون عدد المكونات مُحددًا بعدد
العينات. تحتوي العديد من مجموعات البيانات في العالم الحقيقي على عدد كبير من العينات! في
هذه الحالات، يكون إيجاد *جميع* المكونات باستخدام kPCA الكامل مضيعة لـ
وقت الحساب، حيث يتم وصف البيانات في الغالب بواسطة المكونات القليلة الأولى
(على سبيل المثال ``n_components<=100``). بمعنى آخر، مصفوفة غرام المُتمركزة
التي يتم تحليلها ذاتيًا في عملية ملاءمة Kernel PCA لها رتبة فعالة
أصغر بكثير من حجمها. هذه حالة حيث يمكن أن تُوفر
محللات القيم الذاتية التقريبية تسريعًا مع فقدان دقة منخفض جدًا.


.. dropdown:: محللات القيم الذاتية

    يمكن استخدام المعلمة الاختيارية ``eigen_solver='randomized'`` لـ
    *تقليل* وقت الحساب بشكل كبير عندما يكون عدد ``n_components`` المطلوبة
    صغيرًا مُقارنةً بعدد العينات. يعتمد على
    أساليب التحليل العشوائي لإيجاد حل تقريبي في وقت أقصر.

    التعقيد الزمني لـ :class:`KernelPCA` العشوائي هو
    :math:`O(n_{\mathrm{samples}}^2 \cdot n_{\mathrm{components}})`
    بدلاً من :math:`O(n_{\mathrm{samples}}^3)` للأسلوب الدقيق
    المُطبق باستخدام ``eigen_solver='dense'``.

    تتناسب بصمة الذاكرة لـ :class:`KernelPCA` العشوائي أيضًا مع
    :math:`2 \cdot n_{\mathrm{samples}} \cdot n_{\mathrm{components}}` بدلاً من
    :math:`n_{\mathrm{samples}}^2` للأسلوب الدقيق.

    ملاحظة: هذه التقنية هي نفس التقنية المُستخدمة في :ref:`RandomizedPCA`.

    بالإضافة إلى المحللين أعلاه، يمكن استخدام ``eigen_solver='arpack'`` كـ
    طريقة بديلة للحصول على تحليل تقريبي. من الناحية العملية، لا يُوفر هذا الأسلوب
    أوقات تنفيذ معقولة إلا عندما يكون عدد المكونات التي يجب إيجادها
    صغيرًا للغاية. يتم تمكينه افتراضيًا عندما يكون العدد المطلوب من
    المكونات أقل من 10 (بشكل صارم) ويكون عدد العينات أكثر من 200
    (بشكل صارم). راجع :class:`KernelPCA` للتفاصيل.

    .. rubric:: المراجع

    * محلل *dense*:
      `وثائق scipy.linalg.eigh
      <https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.eigh.html>`_

    * محلل *randomized*:

      * الخوارزمية 4.3 في
        :arxiv:`"إيجاد الهيكل مع العشوائية: الخوارزميات
        العشوائية لبناء تحليلات مصفوفة تقريبية" <0909.4061>`
        Halko, et al. (2009)

      * :arxiv:`"تطبيق خوارزمية عشوائية
        لتحليل المكونات الرئيسية" <1412.3510>`
        A. Szlam et al. (2014)

    * محلل *arpack*:
      `وثائق scipy.sparse.linalg.eigsh
      <https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.eigsh.html>`_
      R. B. Lehoucq, D. C. Sorensen, and C. Yang, (1998)


.. _LSA:

تحليل القيمة الفردية المقطوع والتحليل الدلالي الكامن
===================================================================

:class:`TruncatedSVD` يُطبق متغيرًا من تحليل القيمة الفردية
(SVD) الذي يحسب فقط أكبر :math:`k` قيمة فردية،
حيث :math:`k` هي معلمة مُحددة من قبل المستخدم.

:class:`TruncatedSVD` مُشابه جدًا لـ :class:`PCA`، ولكنه يختلف
في أن المصفوفة :math:`X` لا تحتاج إلى أن تكون مُتمركزة.
عندما يتم طرح متوسطات :math:`X` حسب الأعمدة (لكل ميزة) من قيم الميزات،
فإن SVD المقطوع على المصفوفة الناتجة يُعادل PCA.

.. dropdown:: حول SVD المقطوع والتحليل الدلالي الكامن (LSA)

    عندما يتم تطبيق SVD المقطوع على مصفوفات مُصطلح-مستند
    (كما تم إرجاعها بواسطة :class:`~sklearn.feature_extraction.text.CountVectorizer` أو
    :class:`~sklearn.feature_extraction.text.TfidfVectorizer`)،
    يُعرف هذا التحويل باسم
    `التحليل الدلالي الكامن <https://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_
    (LSA)، لأنه يُحوِّل هذه المصفوفات
    إلى فضاء "دلالي" ذي أبعاد منخفضة.
    على وجه الخصوص، من المعروف أن LSA يُكافح آثار الترادف وتعدد المعاني
    (كلاهما يعني تقريبًا أن هناك معاني متعددة لكل كلمة)،
    مما يتسبب في أن تكون مصفوفات مُصطلح-مستند متفرقة بشكل مفرط
    وتُظهر تشابهًا ضعيفًا في ظل مقاييس مثل تشابه جيب التمام.

    .. note::
        يُعرف LSA أيضًا باسم الفهرسة الدلالية الكامنة، LSI،
        على الرغم من أن ذلك يشير بدقة إلى استخدامه في الفهارس الدائمة
        لأغراض استرجاع المعلومات.

    رياضيًا، ينتج SVD المقطوع المُطبق على عينات التدريب :math:`X`
    تقريبًا منخفض الرتبة :math:`X`:

    .. math::
        X \approx X_k = U_k \Sigma_k V_k^\top

    بعد هذه العملية، :math:`U_k \Sigma_k`
    هي مجموعة التدريب المُحوَّلة بـ :math:`k` ميزات
    (تُسمى ``n_components`` في API).

    لتحويل مجموعة اختبار :math:`X` أيضًا، نضربها في :math:`V_k`:

    .. math::
        X' = X V_k

    .. note::
        معظم مُعالجات LSA في أدبيات مُعالجة اللغة الطبيعية (NLP)
        واسترجاع المعلومات (IR)
        تُبدِّل محاور المصفوفة :math:`X` بحيث يكون لها شكل
        ``(n_features, n_samples)``.
        نُقدم LSA بطريقة مختلفة تتوافق مع واجهة برمجة تطبيقات scikit-learn بشكل أفضل،
        لكن القيم الفردية التي تم العثور عليها هي نفسها.

    بينما يعمل مُحوِّل :class:`TruncatedSVD`
    مع أي مصفوفة ميزات،
    يُوصى باستخدامه على مصفوفات tf-idf على حسابات التردد الأولية
    في إعداد LSA / مُعالجة المستندات.
    على وجه الخصوص، يجب تشغيل تغيير المقياس الفرعي وتردد المستند العكسي
    (``sublinear_tf=True, use_idf=True``)
    لتقريب قيم الميزات من التوزيع الغاوسي،
    للتعويض عن افتراضات LSA الخاطئة حول البيانات النصية.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_text_plot_document_clustering.py`

.. rubric:: المراجع

* Christopher D. Manning, Prabhakar Raghavan and Hinrich Schütze (2008),
  *مقدمة في استرجاع المعلومات*, Cambridge University Press,
  الفصل 18: `تحليلات المصفوفة والفهرسة الدلالية الكامنة
  <https://nlp.stanford.edu/IR-book/pdf/18lsi.pdf>`_



.. _DictionaryLearning:

تعلم القاموس
===================

.. _SparseCoder:

الترميز المتفرق بقاموس مُسبق الحساب
-------------------------------------------

كائن :class:`SparseCoder` هو مُقدِّر يمكن استخدامه لتحويل الإشارات
إلى مجموعات خطية متفرقة من الذرات من قاموس ثابت مُسبق الحساب
مثل أساس المويجات المنفصلة. لذلك لا يُطبق هذا الكائن
أسلوب ``fit``. يُعادل التحويل
مشكلة ترميز متفرقة: إيجاد تمثيل للبيانات كمجموعة خطية من
أقل عدد ممكن من ذرات القاموس. تُطبق جميع أشكال
تعلم القاموس أساليب التحويل التالية، التي يمكن التحكم فيها عبر
معلمة التهيئة ``transform_method``:

* مطاردة التطابق المتعامد (:ref:`omp`)

* انحدار الزاوية الصغرى (:ref:`least_angle_regression`)

* Lasso محسوب بواسطة انحدار الزاوية الصغرى

* Lasso باستخدام النزول الإحداثي (:ref:`lasso`)

* تحديد العتبة

تحديد العتبة سريع جدًا ولكنه لا يُعطي عمليات إعادة بناء دقيقة.
لقد ثبت أنها مفيدة في الأدبيات لمهام التصنيف. بالنسبة لمهام
إعادة بناء الصور، تُعطي مطاردة التطابق المتعامد إعادة البناء الأكثر دقة
وغير المُتحيزة.

تُوفر كائنات تعلم القاموس، عبر المعلمة ``split_code``،
إمكانية فصل القيم الموجبة والسالبة في نتائج
الترميز المتفرق. هذا مفيد عندما يُستخدم تعلم القاموس لاستخراج
الميزات التي سيتم استخدامها للتعلم الخاضع للإشراف، لأنه يسمح لـ
خوارزمية التعلم بتعيين أوزان مختلفة للأحمال السالبة لـ
ذرة مُعينة، من التحميل الموجب المُقابل.

كود التقسيم لعينة واحدة له طول ``2 * n_components``
ويتم إنشاؤه باستخدام القاعدة التالية: أولاً، يتم حساب الكود العادي بطول
``n_components``. ثم، يتم ملء أول ``n_components`` إدخال لـ ``split_code``
بجزء موجب من متجه الكود العادي. النصف الثاني من
كود التقسيم ممتلئ بالجزء السالب من متجه الكود، فقط مع
إشارة موجبة. لذلك، فإن split_code غير سالب.


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_sparse_coding.py`


تعلم القاموس العام
---------------------------

تعلم القاموس (:class:`DictionaryLearning`) هو مشكلة تحليل مصفوفة
يُعادل إيجاد قاموس (عادةً ما يكون مكتملًا) سيؤدي
أداءً جيدًا في الترميز المتفرق للبيانات المُناسبة.

تم اقتراح تمثيل البيانات كمجموعات متفرقة من الذرات من
قاموس مكتمل ليكون الطريقة التي يعمل بها القشرة البصرية الأولية للثدييات.
ونتيجة لذلك، فقد ثبت أن تعلم القاموس المُطبق على بقع الصور
يُعطي نتائج جيدة في مهام معالجة الصور مثل إكمال الصور و
الإصلاح وإزالة الضوضاء، وكذلك لمهام التعرف الخاضعة للإشراف.

تعلم القاموس هو مشكلة تحسين يتم حلها عن طريق التحديث بالتناوب
للكود المتفرق، كحل لمشاكل Lasso متعددة، مع الأخذ في الاعتبار
القاموس ثابتًا، ثم تحديث القاموس ليناسب الكود المتفرق بشكل أفضل.

.. math::
   (U^*, V^*) = \underset{U, V}{\operatorname{arg\,min\,}} & \frac{1}{2}
                ||X-UV||_{\text{Fro}}^2+\alpha||U||_{1,1} \\
                \text{subject to } & ||V_k||_2 <= 1 \text{ for all }
                0 \leq k < n_{\mathrm{atoms}}


.. |pca_img2| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. |dict_img2| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_007.png
   :target: ../auto_examples/decomposition/plot_faces_decomposition.html
   :scale: 60%

.. centered:: |pca_img2| |dict_img2|

:math:`||.||_{\text{Fro}}` تعني قاعدة Frobenius و :math:`||.||_{1,1}`
تعني قاعدة المصفوفة حسب الإدخال وهي مجموع القيم المطلقة
لجميع الإدخالات في المصفوفة.
بعد استخدام هذا الإجراء لملاءمة القاموس، يكون التحويل ببساطة
خطوة ترميز متفرقة تشترك في نفس التطبيق مع جميع
كائنات تعلم القاموس (انظر :ref:`SparseCoder`).

من الممكن أيضًا تقييد القاموس و / أو الكود ليكون موجبًا لـ
مطابقة القيود التي قد تكون موجودة في البيانات. فيما يلي الوجوه مع
قيود الإيجابية المختلفة المُطبقة. يشير اللون الأحمر إلى القيم السالبة، والأزرق
يشير إلى القيم الموجبة، والأبيض يُمثِّل الأصفار.


.. |dict_img_pos1| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_010.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :scale: 60%

.. |dict_img_pos2| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_011.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :scale: 60%

.. |dict_img_pos3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_012.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :scale: 60%

.. |dict_img_pos4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_013.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :scale: 60%

.. centered:: |dict_img_pos1| |dict_img_pos2|
.. centered:: |dict_img_pos3| |dict_img_pos4|


تُظهر الصورة التالية كيف يبدو القاموس الذي تم تعلمه من بقع صور 4x4 بكسل
مستخرجة من جزء من صورة وجه راكون.


.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_image_denoising_001.png
    :target: ../auto_examples/decomposition/plot_image_denoising.html
    :align: center
    :scale: 50%


.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_image_denoising.py`


.. rubric:: المراجع

* `"تعلم القاموس على الإنترنت للترميز المتفرق"
  <https://www.di.ens.fr/sierra/pdfs/icml09.pdf>`_
  J. Mairal, F. Bach, J. Ponce, G. Sapiro, 2009

.. _MiniBatchDictionaryLearning:

تعلم القاموس الصغير الدفعي
------------------------------

:class:`MiniBatchDictionaryLearning` يُطبق إصدارًا أسرع ولكنه أقل دقة
لخوارزمية تعلم القاموس وهو أكثر ملاءمة لمجموعات البيانات
الكبيرة.

افتراضيًا، :class:`MiniBatchDictionaryLearning` يقسم البيانات إلى
دفعات صغيرة ويُحسِّن نموذج NMF بطريقة على الإنترنت عن طريق التدوير على الدفعات الصغيرة
لعدد التكرارات المُحدد. تتحكم معلمة ``batch_size`` في
حجم الدفعات.

من أجل تسريع الخوارزمية الصغيرة الدفعية، من الممكن أيضًا تغيير مقياس
الدفعات السابقة، مما يُعطيها أهمية أقل من الدفعات الأحدث. يتم ذلك
عن طريق إدخال ما يسمى بعامل النسيان الذي تتحكم فيه معلمة ``forget_factor``.

يُطبق المُقدِّر أيضًا ``partial_fit``، الذي يُحدِّث ``H`` عن طريق التكرار
مرة واحدة فقط على دفعة صغيرة. يمكن استخدام هذا للتعلم على الإنترنت عندما لا تكون البيانات
متاحة بسهولة من البداية، أو عندما لا تتناسب البيانات مع الذاكرة.

.. currentmodule:: sklearn.cluster

.. image:: ../auto_examples/cluster/images/sphx_glr_plot_dict_face_patches_001.png
    :target: ../auto_examples/cluster/plot_dict_face_patches.html
    :scale: 50%
    :align: right

.. topic:: **التجميع لتعلم القاموس**

   لاحظ أنه عند استخدام تعلم القاموس لاستخراج تمثيل
   (على سبيل المثال للترميز المتفرق)، يمكن أن يكون التجميع بديلاً جيدًا لتعلم
   القاموس. على سبيل المثال، مُقدِّر :class:`MiniBatchKMeans`
   فعال من الناحية الحسابية ويُطبق التعلم على الإنترنت باستخدام
   أسلوب ``partial_fit``.

   مثال: :ref:`sphx_glr_auto_examples_cluster_plot_dict_face_patches.py`

.. currentmodule:: sklearn.decomposition

.. _FA:

تحليل العوامل
===============

في التعلم غير الخاضع للإشراف، لدينا فقط مجموعة بيانات :math:`X = \{x_1, x_2, \dots, x_n
\}`. كيف يمكن وصف مجموعة البيانات هذه رياضيًا؟ نموذج
`متغير كامن مُستمر` بسيط جدًا لـ :math:`X` هو

.. math:: x_i = W h_i + \mu + \epsilon

يُسمى المتجه :math:`h_i` "كامنًا" لأنه غير مُلاحظ. :math:`\epsilon` يعتبر
مُصطلح ضوضاء موزع وفقًا لتوزيع غاوسي بمتوسط 0 و
تغاير مشترك :math:`\Psi` (أي :math:`\epsilon \sim \mathcal{N}(0, \Psi)`), :math:`\mu` هو
متجه إزاحة عشوائي. يُسمى هذا النموذج "تكويني" لأنه يصف
كيف يتم إنشاء :math:`x_i` من :math:`h_i`. إذا استخدمنا جميع :math:`x_i` كأعمدة لتشكيل
مصفوفة :math:`\mathbf{X}` وجميع :math:`h_i` كأعمدة لمصفوفة :math:`\mathbf{H}`
فيمكننا الكتابة (مع :math:`\mathbf{M}` و :math:`\mathbf{E}` مُحددين بشكل مناسب):

.. math::
    \mathbf{X} = W \mathbf{H} + \mathbf{M} + \mathbf{E}

بمعنى آخر، قمنا *بتحليل* المصفوفة :math:`\mathbf{X}`.

إذا تم إعطاء :math:`h_i`، فإن المعادلة أعلاه تُشير تلقائيًا إلى التفسير
الاحتمالي التالي:

.. math:: p(x_i|h_i) = \mathcal{N}(Wh_i + \mu, \Psi)

بالنسبة لنموذج احتمالي كامل، نحتاج أيضًا إلى توزيع مُسبق لـ
المتغير الكامن :math:`h`. الافتراض الأكثر وضوحًا (بناءً على الخصائص الجيدة
لتوزيع غاوسي) هو :math:`h \sim \mathcal{N}(0,
\mathbf{I})`. هذا يُعطي توزيع غاوسي كتوزيع هامشي لـ :math:`x`:

.. math:: p(x) = \mathcal{N}(\mu, WW^T + \Psi)

الآن، بدون أي افتراضات أخرى، ستكون فكرة وجود متغير كامن :math:`h`
غير ضرورية - يمكن نمذجة :math:`x` بالكامل بمتوسط
وتغاير مشترك. نحتاج إلى فرض بعض الهياكل الأكثر تحديدًا على واحد
من هاتين المعلمتين. افتراض إضافي بسيط يتعلق بـ
هيكل تغاير الخطأ :math:`\Psi`:

* :math:`\Psi = \sigma^2 \mathbf{I}`: يؤدي هذا الافتراض إلى
  النموذج الاحتمالي لـ :class:`PCA`.

* :math:`\Psi = \mathrm{diag}(\psi_1, \psi_2, \dots, \psi_n)`: يُسمى هذا النموذج
  :class:`FactorAnalysis`، وهو نموذج إحصائي كلاسيكي. تُسمى المصفوفة W أحيانًا
  "مصفوفة تحميل العامل".

يُقدِّر كلا النموذجين بشكل أساسي توزيع غاوسي مع مصفوفة تغاير مشترك منخفضة الرتبة.
نظرًا لأن كلا النموذجين احتماليان، فيمكن دمجهما في نماذج أكثر تعقيدًا،
على سبيل المثال خليط من مُحللي العوامل. يحصل المرء على نماذج مُختلفة جدًا (على سبيل المثال
:class:`FastICA`) إذا تم افتراض توزيعات مُسبقة غير غاوسية على المتغيرات الكامنة.

*يمكن* لتحليل العوامل إنتاج مكونات مُشابهة (أعمدة مصفوفة التحميل الخاصة به)
لـ :class:`PCA`. ومع ذلك، لا يمكن للمرء إصدار أي بيانات عامة
حول هذه المكونات (على سبيل المثال ما إذا كانت متعامدة):

.. |pca_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |fa_img3| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_008.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img3| |fa_img3|

الميزة الرئيسية لتحليل العوامل على :class:`PCA` هي أنه
يمكنه نمذجة التباين في كل اتجاه من فضاء الإدخال بشكل مستقل
(ضوضاء غير متجانسة):

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_009.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :align: center
    :scale: 75%

هذا يسمح باختيار نموذج أفضل من PCA الاحتمالي في وجود
ضوضاء غير متجانسة:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_pca_vs_fa_model_selection_002.png
    :target: ../auto_examples/decomposition/plot_pca_vs_fa_model_selection.html
    :align: center
    :scale: 75%

غالبًا ما يتبع تحليل العوامل دوران العوامل (مع
المعلمة `rotation`)، عادةً لتحسين القابلية للتفسير. على سبيل المثال،
يُعظِّم دوران Varimax مجموع تباينات الأحمال التربيعية،
أي أنه يميل إلى إنتاج عوامل أكثر تفرقًا، والتي تتأثر ببضع
ميزات فقط لكل منها ("الهيكل البسيط"). انظر على سبيل المثال المثال الأول أدناه.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_varimax_fa.py`
* :ref:`sphx_glr_auto_examples_decomposition_plot_pca_vs_fa_model_selection.py`


.. _ICA:

تحليل المكونات المستقلة (ICA)
====================================

يفصل تحليل المكونات المستقلة إشارة متعددة المتغيرات إلى
مكونات فرعية مضافة مستقلة إلى أقصى حد. يتم
تطبيقه في scikit-learn باستخدام خوارزمية :class:`Fast ICA <FastICA>`.
عادةً، لا يتم استخدام ICA لتقليل الأبعاد ولكن
لفصل الإشارات المتراكبة. نظرًا لأن نموذج ICA لا يتضمن
مُصطلح ضوضاء، لكي يكون النموذج صحيحًا، يجب تطبيق التبييض.
يمكن القيام بذلك داخليًا باستخدام وسيطة whiten أو يدويًا باستخدام أحد
متغيرات PCA.

يُستخدم بشكل كلاسيكي لفصل الإشارات المختلطة (مشكلة تُعرف باسم
*فصل المصدر الأعمى*)، كما في المثال أدناه:

.. figure:: ../auto_examples/decomposition/images/sphx_glr_plot_ica_blind_source_separation_001.png
    :target: ../auto_examples/decomposition/plot_ica_blind_source_separation.html
    :align: center
    :scale: 60%


يمكن أيضًا استخدام ICA كتحليل غير خطي آخر يجد
مكونات ذات بعض التفرق:

.. |pca_img4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |ica_img4| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_004.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img4| |ica_img4|

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_ica_blind_source_separation.py`
* :ref:`sphx_glr_auto_examples_decomposition_plot_ica_vs_pca.py`
* :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`


.. _NMF:

تحليل المصفوفة غير السالبة (NMF أو NNMF)
===============================================

NMF مع قاعدة Frobenius
---------------------------

:class:`NMF` [1]_ هو نهج بديل للتحليل الذي يفترض أن
البيانات والمكونات غير سالبة. يمكن توصيل :class:`NMF`
بدلاً من :class:`PCA` أو متغيراته، في الحالات التي لا تحتوي فيها مصفوفة البيانات
على قيم سالبة. يجد تحليلًا للعينات
:math:`X` إلى مصفوفتين :math:`W` و :math:`H` من العناصر غير السالبة،
عن طريق تحسين المسافة :math:`d` بين :math:`X` وحاصل ضرب المصفوفة
:math:`WH`. دالة المسافة الأكثر استخدامًا هي قاعدة Frobenius
التربيعية، وهي امتداد واضح لقاعدة إقليدية للمصفوفات:

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{\mathrm{Fro}}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

على عكس :class:`PCA`، يتم الحصول على تمثيل المتجه بطريقة مضافة،
عن طريق تراكب المكونات، بدون طرح. هذه النماذج المضافة
فعالة لتمثيل الصور والنصوص.

وقد لوحظ في [Hoyer, 2004] [2]_ أنه، عند تقييده بعناية،
يمكن لـ :class:`NMF` إنتاج تمثيل قائم على الأجزاء لمجموعة البيانات،
مما يؤدي إلى نماذج قابلة للتفسير. يعرض المثال التالي 16
مكونًا متفرقًا عثر عليها :class:`NMF` من الصور في مجموعة بيانات
وجوه Olivetti، مُقارنةً بصور PCA الذاتية.

.. |pca_img5| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_002.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. |nmf_img5| image:: ../auto_examples/decomposition/images/sphx_glr_plot_faces_decomposition_003.png
    :target: ../auto_examples/decomposition/plot_faces_decomposition.html
    :scale: 60%

.. centered:: |pca_img5| |nmf_img5|


تحدد السمة `init` أسلوب التهيئة المُطبق، والذي
له تأثير كبير على أداء الأسلوب. :class:`NMF` يُطبق
أسلوب تحليل القيمة الفردية المزدوج غير السالب. يعتمد NNDSVD [4]_ على
عمليتي SVD، إحداهما تُقارب مصفوفة البيانات، والأخرى تُقارب
أقسامًا موجبة من عوامل SVD الجزئية الناتجة باستخدام خاصية جبرية
لمصفوفات الرتبة الوحدة. خوارزمية NNDSVD الأساسية مُناسبة بشكل أفضل لـ
التحليل المتفرق. يُوصى بمتغيراتها NNDSVDa (حيث يتم تعيين جميع الأصفار
على متوسط جميع عناصر البيانات)، و NNDSVDar (حيث يتم تعيين الأصفار
إلى اضطرابات عشوائية أقل من متوسط البيانات مقسومًا على 100)
في الحالة الكثيفة.

لاحظ أن محلل التحديث الضربي ('mu') لا يمكنه تحديث الأصفار الموجودة في
التهيئة، لذلك يؤدي إلى نتائج أسوأ عند استخدامه مع
خوارزمية NNDSVD الأساسية التي تُقدم الكثير من الأصفار؛ في هذه الحالة،
يجب تفضيل NNDSVDa أو NNDSVDar.

يمكن أيضًا تهيئة :class:`NMF` بمصفوفات عشوائية غير سالبة مُقيَّسة بشكل صحيح
عن طريق تعيين `init="random"`. يمكن أيضًا تمرير عدد صحيح أو
``RandomState`` إلى `random_state` للتحكم في
إمكانية إعادة الإنتاج.

في :class:`NMF`، يمكن إضافة مُسبقات L1 و L2 إلى دالة الخسارة من أجل
تنظيم النموذج. يستخدم مُسبق L2 قاعدة Frobenius، بينما يستخدم مُسبق L1
قاعدة L1 حسب العنصر. كما هو الحال في :class:`~sklearn.linear_model.ElasticNet`،
نتحكم في مزيج L1 و L2 باستخدام معلمة `l1_ratio` (:math:`\rho`)،
وشدة التنظيم باستخدام معلمات `alpha_W` و
`alpha_H` (:math:`\alpha_W` و :math:`\alpha_H`). يتم
تغيير مقياس المُسبقات حسب عدد العينات (:math:`n\_samples`) لـ `H` وعدد
الميزات (:math:`n\_features`) لـ `W` للحفاظ على تأثيرها متوازنًا
مع بعضها البعض ومع مُصطلح ملاءمة البيانات مستقلًا قدر الإمكان عن
حجم مجموعة التدريب. ثم مُصطلحات المُسبقات هي:

.. math::
    (\alpha_W \rho ||W||_1 + \frac{\alpha_W(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2) * n\_features
    + (\alpha_H \rho ||H||_1 + \frac{\alpha_H(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2) * n\_samples

ودالة الهدف المُنظَّمة هي:

.. math::
    d_{\mathrm{Fro}}(X, WH)
    + (\alpha_W \rho ||W||_1 + \frac{\alpha_W(1-\rho)}{2} ||W||_{\mathrm{Fro}} ^ 2) * n\_features
    + (\alpha_H \rho ||H||_1 + \frac{\alpha_H(1-\rho)}{2} ||H||_{\mathrm{Fro}} ^ 2) * n\_samples

NMF مع تباعد بيتا
--------------------------

كما هو موضح سابقًا، فإن دالة المسافة الأكثر استخدامًا هي قاعدة Frobenius
التربيعية، وهي امتداد واضح لقاعدة إقليدية لـ
المصفوفات:

.. math::
    d_{\mathrm{Fro}}(X, Y) = \frac{1}{2} ||X - Y||_{Fro}^2 = \frac{1}{2} \sum_{i,j} (X_{ij} - {Y}_{ij})^2

يمكن استخدام دوال مسافة أخرى في NMF كما هو الحال، على سبيل المثال، تباعد
Kullback-Leibler (KL) (المُعمم)، والذي يُشار إليه أيضًا باسم I-divergence:

.. math::
    d_{KL}(X, Y) = \sum_{i,j} (X_{ij} \log(\frac{X_{ij}}{Y_{ij}}) - X_{ij} + Y_{ij})

أو تباعد Itakura-Saito (IS):

.. math::
    d_{IS}(X, Y) = \sum_{i,j} (\frac{X_{ij}}{Y_{ij}} - \log(\frac{X_{ij}}{Y_{ij}}) - 1)

هذه المسافات الثلاث هي حالات خاصة لعائلة تباعد بيتا، مع
:math:`\beta = 2, 1, 0` على التوالي [6]_. يتم تعريف تباعد بيتا بواسطة:

.. math::
    d_{\beta}(X, Y) = \sum_{i,j} \frac{1}{\beta(\beta - 1)}(X_{ij}^\beta + (\beta-1)Y_{ij}^\beta - \beta X_{ij} Y_{ij}^{\beta - 1})

.. image:: ../images/beta_divergence.png
    :align: center
    :scale: 75%

لاحظ أن هذا التعريف غير صالح إذا :math:`\beta \in (0; 1)`، ومع ذلك يمكن
تمديده باستمرار إلى تعريفات :math:`d_{KL}` و :math:`d_{IS}`
على التوالي.

.. dropdown:: محللات NMF المُطبقة

    :class:`NMF` يُطبق محللين، باستخدام النزول الإحداثي ('cd') [5]_، و
    التحديث الضربي ('mu') [6]_. يمكن لمحلل 'mu' تحسين كل
    تباعد بيتا، بما في ذلك بالطبع قاعدة Frobenius (:math:`\beta=2`)، و
    تباعد Kullback-Leibler (المُعمم) (:math:`\beta=1`) وتباعد
    Itakura-Saito (:math:`\beta=0`). لاحظ أنه بالنسبة لـ
    :math:`\beta \in (1; 2)`، يكون محلل 'mu' أسرع بكثير من قيم
    :math:`\beta` الأخرى. لاحظ أيضًا أنه مع :math:`\beta` سالبة (أو 0، أي
    'itakura-saito')، لا يمكن أن تحتوي مصفوفة الإدخال على قيم صفرية.

    يمكن لمحلل 'cd' تحسين قاعدة Frobenius فقط. نظرًا لـ
    عدم التحدب الكامن لـ NMF، قد تتقارب المحللات المختلفة إلى
    حدود دنيا مختلفة، حتى عند تحسين نفس دالة المسافة.

من الأفضل استخدام NMF مع أسلوب ``fit_transform``، الذي يُعيد المصفوفة W.
يتم تخزين المصفوفة H في النموذج المُناسب في الخاصية ``components_``؛
سيقوم الأسلوب ``transform`` بتحليل مصفوفة جديدة X_new بناءً على هذه
المكونات المخزنة::

    >>> import numpy as np
    >>> X = np.array([[1, 1], [2, 1], [3, 1.2], [4, 1], [5, 0.8], [6, 1]])
    >>> from sklearn.decomposition import NMF
    >>> model = NMF(n_components=2, init='random', random_state=0)
    >>> W = model.fit_transform(X)
    >>> H = model.components_
    >>> X_new = np.array([[1, 0], [1, 6.1], [1, 0], [1, 4], [3.2, 1], [0, 4]])
    >>> W_new = model.transform(X_new)



.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_decomposition_plot_faces_decomposition.py`
* :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`

.. _MiniBatchNMF:

تحليل المصفوفة غير السالبة الصغير الدفعي
--------------------------------------------

:class:`MiniBatchNMF` [7]_ يُطبق إصدارًا أسرع ولكنه أقل دقة
لتحليل المصفوفة غير السالبة (أي :class:`~sklearn.decomposition.NMF`)،
مُناسب بشكل أفضل لمجموعات البيانات الكبيرة.

افتراضيًا، :class:`MiniBatchNMF` يقسم البيانات إلى دفعات صغيرة و
يُحسِّن نموذج NMF بطريقة على الإنترنت عن طريق التدوير على الدفعات الصغيرة
لعدد التكرارات المُحدد. تتحكم معلمة ``batch_size`` في
حجم الدفعات.

من أجل تسريع الخوارزمية الصغيرة الدفعية، من الممكن أيضًا تغيير مقياس
الدفعات السابقة، مما يُعطيها أهمية أقل من الدفعات الأحدث. يتم ذلك
بإدخال ما يسمى بعامل النسيان الذي تتحكم فيه معلمة ``forget_factor``.

يُطبق المُقدِّر أيضًا ``partial_fit``، الذي يُحدِّث ``H`` عن طريق التكرار
مرة واحدة فقط على دفعة صغيرة. يمكن استخدام هذا للتعلم على الإنترنت عندما لا تكون البيانات
متاحة بسهولة من البداية، أو عندما لا تتناسب البيانات مع الذاكرة.

.. rubric:: المراجع

.. [1] `"تعلم أجزاء الكائنات عن طريق تحليل المصفوفة غير السالبة"
  <http://www.cs.columbia.edu/~blei/fogm/2020F/readings/LeeSeung1999.pdf>`_
  D. Lee, S. Seung, 1999

.. [2] `"تحليل المصفوفة غير السالبة مع قيود التفرق"
  <https://www.jmlr.org/papers/volume5/hoyer04a/hoyer04a.pdf>`_
  P. Hoyer, 2004

.. [4] `"التهيئة القائمة على SVD: بداية جيدة لتحليل
  المصفوفة غير السالبة"
  <https://www.boutsidis.org/Boutsidis_PRE_08.pdf>`_
  C. Boutsidis, E. Gallopoulos, 2008

.. [5] `"خوارزميات محلية سريعة لتحليل المصفوفة والموتر غير السالب
  واسع النطاق."
  <https://www.researchgate.net/profile/Anh-Huy-Phan/publication/220241471_Fast_Local_Algorithms_for_Large_Scale_Nonnegative_Matrix_and_Tensor_Factorizations>`_
  A. Cichocki, A. Phan, 2009

.. [6] :arxiv:`"خوارزميات لتحليل المصفوفة غير السالبة مع
  تباعد بيتا" <1010.1763>`
  C. Fevotte, J. Idier, 2011

.. [7] :arxiv:`"خوارزميات على الإنترنت لتحليل المصفوفة غير السالبة مع
  تباعد Itakura-Saito" <1106.4198>`
  A. Lefevre, F. Bach, C. Fevotte, 2011

.. _LatentDirichletAllocation:

تخصيص ديريتشليت الكامن (LDA)
=================================

تخصيص ديريتشليت الكامن هو نموذج احتمالي تكويني لمجموعات
مجموعات البيانات المنفصلة مثل مجموعات النصوص. وهو أيضًا نموذج موضوع يُستخدم لـ
اكتشاف مواضيع مُجردة من مجموعة من المستندات.

النموذج الرسومي لـ LDA هو نموذج تكويني من ثلاثة مستويات:

.. image:: ../images/lda_model_graph.png
   :align: center

ملاحظة حول الرموز المُقدمة في النموذج الرسومي أعلاه، والتي يمكن العثور عليها في
Hoffman et al. (2013):

* المجموعة هي مجموعة من :math:`D` مستندات.
* المستند هو تسلسل من :math:`N` كلمات.
* هناك :math:`K` مواضيع في المجموعة.
* تُمثِّل المربعات أخذ عينات متكرر.

في النموذج الرسومي، كل عقدة هي متغير عشوائي ولها دور في
عملية التكوين. تشير العقدة المُظللة إلى متغير مُلاحظ وتشير العقدة
غير المُظللة إلى متغير خفي (كامن). في هذه الحالة، الكلمات في المجموعة هي
البيانات الوحيدة التي نُلاحظها. تحدد المتغيرات الكامنة الخليط العشوائي
للمواضيع في المجموعة وتوزيع الكلمات في المستندات.
الهدف من LDA هو استخدام الكلمات المُلاحظة للاستدلال على هيكل الموضوع
الخفي.

.. dropdown:: تفاصيل حول نمذجة مجموعات النصوص

    عند نمذجة مجموعات النصوص، يفترض النموذج عملية التكوين التالية
    لمجموعة تحتوي على :math:`D` مستندات و :math:`K` مواضيع، مع :math:`K`
    مُقابلة لـ `n_components` في API:

    1. لكل موضوع :math:`k \in K`، ارسم :math:`\beta_k \sim
       \mathrm{Dirichlet}(\eta)`. هذا يُوفر توزيعًا على الكلمات،
       أي احتمال ظهور كلمة في الموضوع :math:`k`.
       :math:`\eta` تقابل `topic_word_prior`.

    2. لكل مستند :math:`d \in D`، ارسم نسب الموضوع
       :math:`\theta_d \sim \mathrm{Dirichlet}(\alpha)`. :math:`\alpha`
       تقابل `doc_topic_prior`.

    3. لكل كلمة :math:`i` في المستند :math:`d`:

       a. ارسم تعيين الموضوع :math:`z_{di} \sim \mathrm{Multinomial}
          (\theta_d)`
       b. ارسم الكلمة المُلاحظة :math:`w_{ij} \sim \mathrm{Multinomial}
          (\beta_{z_{di}})`

    لتقدير المعلمات، التوزيع اللاحق هو:

    .. math::
        p(z, \theta, \beta |w, \alpha, \eta) =
        \frac{p(z, \theta, \beta|\alpha, \eta)}{p(w|\alpha, \eta)}

    نظرًا لأن التوزيع اللاحق صعب التعامل معه، فإن أسلوب بايز المتغير
    يستخدم توزيعًا أبسط :math:`q(z,\theta,\beta | \lambda, \phi, \gamma)`
    لتقريبه، ويتم تحسين معلمات التباين هذه :math:`\lambda` و
    :math:`\phi` و :math:`\gamma` لزيادة الحد الأدنى للأدلة
    (ELBO):

    .. math::
        \log\: P(w | \alpha, \eta) \geq L(w,\phi,\gamma,\lambda) \overset{\triangle}{=}
        E_{q}[\log\:p(w,z,\theta,\beta|\alpha,\eta)] - E_{q}[\log\:q(z, \theta, \beta)]

    زيادة ELBO يُعادل تقليل تباعد Kullback-Leibler (KL)
    بين :math:`q(z,\theta,\beta)` والتوزيع اللاحق الحقيقي
    :math:`p(z, \theta, \beta |w, \alpha, \eta)`.


:class:`LatentDirichletAllocation` يُطبق خوارزمية بايز المتغيرة على الإنترنت و
يدعم كل من أساليب التحديث على الإنترنت والدفعية.
بينما يُحدِّث الأسلوب الدفعي متغيرات التباين بعد كل تمريرة كاملة عبر
البيانات، يُحدِّث الأسلوب على الإنترنت متغيرات التباين من نقاط بيانات
الدفعة الصغيرة.

.. note::

  على الرغم من أن الأسلوب على الإنترنت مُضمَّن للتقارب إلى نقطة مثلى محليًا، إلا أن جودة
  النقطة المثلى وسرعة التقارب قد تعتمد على حجم الدفعة الصغيرة و
  السمات المتعلقة بإعداد مُعدل التعلم.

عندما يتم تطبيق :class:`LatentDirichletAllocation` على مصفوفة "مصطلح-مستند"، سيتم تحليل
المصفوفة إلى مصفوفة "موضوع-مصطلح" ومصفوفة "مستند-موضوع". بينما
يتم تخزين مصفوفة "موضوع-مصطلح" كـ `components_` في النموذج، يمكن حساب مصفوفة
"مستند-موضوع" من أسلوب ``transform``.

:class:`LatentDirichletAllocation` يُطبق أيضًا أسلوب ``partial_fit``. يُستخدم هذا
عندما يمكن جلب البيانات بالتسلسل.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_applications_plot_topics_extraction_with_nmf_lda.py`

.. rubric:: المراجع

* `"تخصيص ديريتشليت الكامن"
  <https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf>`_
  D. Blei, A. Ng, M. Jordan, 2003

* `"التعلم على الإنترنت لتخصيص ديريتشليت الكامن”
  <https://papers.nips.cc/paper/3902-online-learning-for-latent-dirichlet-allocation.pdf>`_
  M. Hoffman, D. Blei, F. Bach, 2010

* `"الاستدلال العشوائي المتغير"
  <https://www.cs.columbia.edu/~blei/papers/HoffmanBleiWangPaisley2013.pdf>`_
  M. Hoffman, D. Blei, C. Wang, J. Paisley, 2013

* `"معيار varimax للدوران التحليلي في تحليل العوامل"
  <https://link.springer.com/article/10.1007%2FBF02289233>`_
  H. F. Kaiser, 1958

انظر أيضًا :ref:`nca_dim_reduction` لتقليل الأبعاد باستخدام
تحليل مكونات الجوار.



