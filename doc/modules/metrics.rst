
.. _metrics:

مقاييس المقارنة الزوجية، والصلات، والنوى (Kernels)
=====================================================

تُطبق الوحدة الفرعية :mod:`sklearn.metrics.pairwise` أدوات مساعدة لتقييم المسافات الزوجية أو صلة مجموعات العينات.

تحتوي هذه الوحدة على كل من مقاييس المسافة والنوى. يتم إعطاء ملخص موجز عن الاثنين هنا.

مقاييس المسافة هي دوال ``d(a, b)`` بحيث ``d(a, b) < d(a, c)`` إذا اعتُبرت الكائنات ``a`` و ``b`` "أكثر تشابهًا" من الكائنات ``a`` و ``c``. سيكون للكائنين المتشابهين تمامًا مسافة صفر. أحد الأمثلة الأكثر شيوعًا هو المسافة الإقليدية. لكي تكون مقياسًا "حقيقيًا"، يجب أن تلتزم بالشروط الأربعة التالية::

    1. d(a, b) >= 0, لجميع a و b
    2. d(a, b) == 0, إذا وفقط إذا a = b, تحديد موجب
    3. d(a, b) == d(b, a), تناظر
    4. d(a, c) <= d(a, b) + d(b, c), متباينة المثلث

النوى هي مقاييس التشابه، أي ``s(a, b) > s(a, c)`` إذا اعتُبرت الكائنات ``a`` و ``b`` "أكثر تشابهًا" من الكائنات ``a`` و ``c``. يجب أن تكون النواة أيضًا شبه محددة موجبة.

هناك عدد من الطرق للتحويل بين مقياس المسافة ومقياس التشابه، مثل النواة. دع ``D`` تكون المسافة، و ``S`` تكون النواة:

1. ``S = np.exp(-D * gamma)``, حيث إحدى الطرق التجريبية لاختيار ``gamma`` هي ``1 / num_features``
2. ``S = 1. / (D / np.max(D))``


.. currentmodule:: sklearn.metrics

يمكن تقييم المسافات بين متجهات الصفوف لـ ``X`` ومتجهات الصفوف لـ ``Y`` باستخدام :func:`pairwise_distances`. إذا تم حذف ``Y``، يتم حساب المسافات الزوجية لمتجهات صفوف ``X``. وبالمثل، يمكن استخدام :func:`pairwise.pairwise_kernels` لحساب النواة بين ``X`` و ``Y`` باستخدام دوال نواة مختلفة. راجع مرجع API لمزيد من التفاصيل.

    >>> import numpy as np
    >>> from sklearn.metrics import pairwise_distances
    >>> from sklearn.metrics.pairwise import pairwise_kernels
    >>> X = np.array([[2, 3], [3, 5], [5, 8]])
    >>> Y = np.array([[1, 0], [2, 1]])
    >>> pairwise_distances(X, Y, metric='manhattan')
    array([[ 4.,  2.],
           [ 7.,  5.],
           [12., 10.]])
    >>> pairwise_distances(X, metric='manhattan')
    array([[0., 3., 8.],
           [3., 0., 5.],
           [8., 5., 0.]])
    >>> pairwise_kernels(X, Y, metric='linear')
    array([[ 2.,  7.],
           [ 3., 11.],
           [ 5., 18.]])


.. currentmodule:: sklearn.metrics.pairwise

.. _cosine_similarity:

تشابه جيب التمام
-----------------
تحسب :func:`cosine_similarity` حاصل الضرب النقطي المعياري L2 للمتجهات. أي، إذا كان :math:`x` و :math:`y` متجهي صفوف، يتم تعريف تشابه جيب التمام :math:`k` على النحو التالي:

.. math::

    k(x, y) = \frac{x y^\top}{\|x\| \|y\|}

يسمى هذا تشابه جيب التمام، لأن التطبيع الإقليدي (L2) يُسقط المتجهات على كرة الوحدة، وحاصل الضرب النقطي هو جيب تمام الزاوية بين النقاط التي تشير إليها المتجهات.

تُعد هذه النواة خيارًا شائعًا لحساب تشابه المستندات الممثلة كمتجهات tf-idf. تقبل :func:`cosine_similarity` مصفوفات ``scipy.sparse``. (لاحظ أن وظيفة tf-idf في ``sklearn.feature_extraction.text`` يمكن أن تنتج متجهات معيارية، وفي هذه الحالة تكون :func:`cosine_similarity` مكافئة لـ :func:`linear_kernel`، ولكنها أبطأ فقط.)

.. rubric:: المراجع

* C.D. Manning, P. Raghavan and H. Schütze (2008). Introduction to
  Information Retrieval. Cambridge University Press.
  https://nlp.stanford.edu/IR-book/html/htmledition/the-vector-space-model-for-scoring-1.html

.. _linear_kernel:

النواة الخطية
-------------
تحسب الدالة :func:`linear_kernel` النواة الخطية، أي حالة خاصة من :func:`polynomial_kernel` مع ``degree=1`` و ``coef0=0`` (متجانسة). إذا كان ``x`` و ``y`` متجهين عموديين، فإن نواتهما الخطية هي:

.. math::

    k(x, y) = x^\top y

.. _polynomial_kernel:

النواة متعددة الحدود
-----------------
تحسب الدالة :func:`polynomial_kernel` النواة متعددة الحدود من الدرجة d بين متجهين. تُمثل النواة متعددة الحدود التشابه بين متجهين. من الناحية المفاهيمية، لا تأخذ النوى متعددة الحدود في الاعتبار التشابه بين المتجهات ضمن نفس البعد فحسب، بل أيضًا عبر الأبعاد. عند استخدامها في خوارزميات تعلم الآلة، يسمح هذا بحساب تفاعل الميزات.

يتم تعريف النواة متعددة الحدود على النحو التالي:

.. math::

    k(x, y) = (\gamma x^\top y +c_0)^d

حيث:

* ``x``, ``y`` هي متجهات الإدخال
* ``d`` هي درجة النواة

إذا كان :math:`c_0 = 0`، يُقال إن النواة متجانسة.

.. _sigmoid_kernel:

النواة السينية (Sigmoid)
--------------
تحسب الدالة :func:`sigmoid_kernel` النواة السينية بين متجهين. تُعرف النواة السينية أيضًا باسم الظل الزائدي، أو بيرسيبترون متعدد الطبقات (لأنه، في مجال الشبكة العصبية، غالبًا ما يتم استخدامه كدالة تنشيط الخلايا العصبية). يتم تعريفها على النحو التالي:

.. math::

    k(x, y) = \tanh( \gamma x^\top y + c_0)

حيث:

* ``x``, ``y`` هي متجهات الإدخال
* :math:`\gamma` يُعرف بالميل
* :math:`c_0` يُعرف بالتقاطع

.. _rbf_kernel:

نواة دالة أساس شعاعي (RBF)
----------
تحسب الدالة :func:`rbf_kernel` نواة دالة الأساس الشعاعي (RBF) بين متجهين. يتم تعريف هذه النواة على النحو التالي:

.. math::

    k(x, y) = \exp( -\gamma \| x-y \|^2)

حيث ``x`` و ``y`` هما متجهات الإدخال. إذا كان :math:`\gamma = \sigma^{-2}`، تُعرف النواة باسم النواة الغاوسية للتباين :math:`\sigma^2`.

.. _laplacian_kernel:

نواة لابلاس
----------------
الدالة :func:`laplacian_kernel` هي متغير على نواة دالة الأساس الشعاعي المُعرَّفة على النحو التالي:

.. math::

    k(x, y) = \exp( -\gamma \| x-y \|_1)

حيث ``x`` و ``y`` هما متجهات الإدخال و :math:`\|x-y\|_1` هي مسافة مانهاتن بين متجهات الإدخال.

لقد أثبتت جدواها في تعلم الآلة المطبق على البيانات غير المزعجة. انظر على سبيل المثال `Machine learning for quantum mechanics in a nutshell
<https://onlinelibrary.wiley.com/doi/10.1002/qua.24954/abstract/>`_.

.. _chi2_kernel:

نواة مربع كاي
------------------
تُعد نواة مربع كاي خيارًا شائعًا جدًا لتدريب SVM غير الخطية في تطبيقات رؤية الكمبيوتر. يمكن حسابها باستخدام :func:`chi2_kernel` ثم تمريرها إلى :class:`~sklearn.svm.SVC` مع ``kernel="precomputed"``::

    >>> from sklearn.svm import SVC
    >>> from sklearn.metrics.pairwise import chi2_kernel
    >>> X = [[0, 1], [1, 0], [.2, .8], [.7, .3]]
    >>> y = [0, 1, 0, 1]
    >>> K = chi2_kernel(X, gamma=.5)
    >>> K
    array([[1.        , 0.36787944, 0.89483932, 0.58364548],
           [0.36787944, 1.        , 0.51341712, 0.83822343],
           [0.89483932, 0.51341712, 1.        , 0.7768366 ],
           [0.58364548, 0.83822343, 0.7768366 , 1.        ]])

    >>> svm = SVC(kernel='precomputed').fit(K, y)
    >>> svm.predict(K)
    array([0, 1, 0, 1])

يمكن أيضًا استخدامها مباشرة كوسيطة ``kernel``::

    >>> svm = SVC(kernel=chi2_kernel).fit(X, y)
    >>> svm.predict(X)
    array([0, 1, 0, 1])


يتم إعطاء نواة مربع كاي بواسطة

.. math::

        k(x, y) = \exp \left (-\gamma \sum_i \frac{(x[i] - y[i]) ^ 2}{x[i] + y[i]} \right )

يفترض أن تكون البيانات غير سالبة، وغالبًا ما يتم تطبيعها للحصول على معيار L1 يساوي واحدًا. يتم ترشيد التطبيع مع الاتصال بمسافة مربع كاي، وهي مسافة بين توزيعات احتمالية منفصلة.

غالبًا ما يتم استخدام نواة مربع كاي في الرسوم البيانية (أكياس) الكلمات المرئية.

.. rubric:: المراجع

* Zhang, J. and Marszalek, M. and Lazebnik, S. and Schmid, C.
  Local features and kernels for classification of texture and object
  categories: A comprehensive study
  International Journal of Computer Vision 2007
  https://hal.archives-ouvertes.fr/hal-00171412/document

