
.. _neural_networks_supervised:

==================================
نماذج الشبكات العصبية (الخاضعة للإشراف)
==================================

.. currentmodule:: sklearn.neural_network


.. warning::

    لا يُقصد بهذا التنفيذ التطبيقات واسعة النطاق. على وجه الخصوص،
    لا يوفر scikit-learn أي دعم لـ GPU. للحصول على تنفيذ أسرع وأكثر فعالية،
    يعتمد على GPU، بالإضافة إلى أطر العمل التي تقدم مرونة أكبر لبناء هندسة التعلم العميق، راجع:  :ref:`related_projects`.

.. _multilayer_perceptron:

متعدد الطبقات Perceptron
======================

**متعدد الطبقات Perceptron (MLP)** هو خوارزمية تعلم خاضعة للإشراف تتعلم
دالة:math:`f: R^m \rightarrow R^o` عن طريق التدريب على مجموعة بيانات،
حيث:math:`m` هو عدد الأبعاد للإدخال و:math:`o` هو
عدد الأبعاد للإخراج. بالنظر إلى مجموعة من الميزات:math:`X = {x_1, x_2, ..., x_m}`
وهدف:math:`y`، يمكنه تعلم دالة تقريب غير خطية إما
للتصنيف أو التراجع. إنه يختلف عن الانحدار اللوجستي، في أنه
بين طبقة الإدخال وطبقة الإخراج، يمكن أن يكون هناك طبقة أو أكثر غير خطية
تسمى الطبقات المخفية. يوضح الشكل 1 شبكة MLP ذات طبقة خفية واحدة مع إخراج قياسي.

.. figure:: ../images/multilayerperceptron_network.png
   :align: center
   :scale: 60%

   **الشكل 1: شبكة MLP ذات طبقة خفية واحدة.**

تتكون الطبقة اليسرى، المعروفة باسم طبقة الإدخال، من مجموعة من العصبونات
:math:`\{x_i | x_1, x_2, ..., x_m\}` التي تمثل ميزات الإدخال. تقوم كل
عصبون في الطبقة المخفية بتحويل القيم من الطبقة السابقة باستخدام
مجموع خطي مرجح:math:`w_1x_1 + w_2x_2 + ... + w_mx_m`، يليه
دالة تنشيط غير خطية:math:`g(\cdot):R \rightarrow R` - مثل
دالة تانغينس الزائدية. تتلقى طبقة الإخراج القيم من
الطبقة الأخيرة وتحولها إلى قيم الإخراج.

يحتوي الوحدة النمطية على السمتين العامتين "coefs_" و"intercepts_".
"coefs_" هي قائمة من مصفوفات الأوزان، حيث تمثل مصفوفة الأوزان عند الفهرس
:math:`i` الأوزان بين الطبقة:math:`i` والطبقة
:math:`i+1`. "intercepts_" هي قائمة من متجهات التحيز، حيث يمثل المتجه
عند الفهرس:math:`i` قيم التحيز المضافة إلى الطبقة:math:`i+1`.

.. dropdown:: مزايا وعيوب متعدد الطبقات Perceptron

  مزايا متعدد الطبقات Perceptron هي:

  + القدرة على تعلم نماذج غير خطية.

  + القدرة على تعلم النماذج في الوقت الفعلي (التعلم عبر الإنترنت)
    باستخدام "partial_fit".


  تتضمن عيوب متعدد الطبقات Perceptron (MLP) ما يلي:

  + MLP ذو الطبقات المخفية له دالة فقدان غير محدبة حيث يوجد
    أكثر من حد أدنى محلي واحد. لذلك يمكن أن تؤدي عمليات التهيئة العشوائية المختلفة للأوزان إلى اختلافات في دقة التحقق.

  + يتطلب MLP ضبط عدد من المعلمات مثل عدد
    العصبونات المخفية، والطبقات، والحلقات.

  + MLP حساس لقياس الميزة.

  يرجى الاطلاع على القسم: ref:`Tips on Practical Use <mlp_tips>` الذي يعالج
بعض هذه العيوب.


التصنيف
==============

تطبق الفئة: class:`MLPClassifier` خوارزمية متعدد الطبقات Perceptron (MLP)
التي تتدرب باستخدام `Backpropagation <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_.

يتدرب MLP على مصفوفتين: مصفوفة X ذات الحجم (n_samples, n_features)، والتي تحتوي
على عينات التدريب الممثلة كمؤشرات قيم النقطة العائمة؛ ومصفوفة
y ذات الحجم (n_samples,)، والتي تحتوي على قيم الهدف (تصنيفات الفئات)
لعينات التدريب::

    >>> from sklearn.neural_network import MLPClassifier
    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    ...                     hidden_layer_sizes=(5, 2), random_state=1)
    ...
    >>> clf.fit(X, y)
    MLPClassifier(alpha=1e-05, hidden_layer_sizes=(5, 2), random_state=1,
                  solver='lbfgs')

بعد التهيئة (التدريب)، يمكن للنموذج التنبؤ بالتصنيفات لعينات جديدة::

    >>> clf.predict([[2., 2.], [-1., -2.]])
    array([1, 0])

يمكن لـ MLP ملاءمة نموذج غير خطي لبيانات التدريب. "clf.coefs_"
يحتوي على مصفوفات الأوزان التي تشكل معلمات النموذج::

    >>> [coef.shape for coef in clf.coefs_]
    [(2, 5), (5, 2), (2, 1)]

حاليًا، :class:`MLPClassifier` يدعم فقط
دالة فقدان Cross-Entropy، والتي تسمح بتقديرات الاحتمالية عن طريق تشغيل
طريقة "predict_proba".

يتدرب MLP باستخدام Backpropagation. بشكل أكثر دقة، يتدرب باستخدام شكل من أشكال
التدرج النازل وتتم حساب التدرجات باستخدام Backpropagation. للتصنيف، فإنه يقلل من دالة فقدان Cross-Entropy، مما يعطي متجهًا
من تقديرات الاحتمالية:math:`P(y|x)` لكل عينة:math:`x`::

    >>> clf.predict_proba([[2., 2.], [1., 2.]])
    array([[1.967...e-04, 9.998...-01],
           [1.967...e-04, 9.998...-01]])

:class:`MLPClassifier` يدعم التصنيف متعدد الفئات من خلال
تطبيق `Softmax <https://en.wikipedia.org/wiki/Softmax_activation_function>`_
كدالة إخراج.

علاوة على ذلك، يدعم النموذج: ref:`multi-label classification <multiclass>`
حيث يمكن أن تنتمي العينة إلى أكثر من فئة واحدة. بالنسبة لكل فئة، يمر الإخراج الخام عبر الدالة اللوغاريتمية. يتم تقريب القيم الأكبر من أو يساوي `0.5`
إلى `1`، وإلا إلى `0`. بالنسبة للإخراج المتوقع لعينة، فإن المؤشرات التي تكون القيمة فيها `1` تمثل الفئات المعينة لتلك العينة::

    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [[0, 1], [1, 1]]
    >>> clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
    ...                     hidden_layer_sizes=(15,), random_state=1)
    ...
    >>> clf.fit(X, y)
    MLPClassifier(alpha=1e-05, hidden_layer_sizes=(15,), random_state=1,
                  solver='lbfgs')
    >>> clf.predict([[1., 2.]])
    array([[1, 1]])
    >>> clf.predict([[0., 0.]])
    array([[0, 1]])

راجع الأمثلة أدناه وdocstring لـ
:meth:`MLPClassifier.fit` لمزيد من المعلومات.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_training_curves.py`
* راجع: ref:`sphx_glr_auto_examples_neural_networks_plot_mnist_filters.py` لل
تمثيل مرئي للأوزان المدربة.

التراجع
==========

تطبق الفئة: class:`MLPRegressor` خوارزمية متعدد الطبقات Perceptron (MLP) التي
تتدرب باستخدام backpropagation بدون دالة تنشيط في طبقة الإخراج،
والتي يمكن اعتبارها أيضًا استخدام دالة الهوية كدالة تنشيط. لذلك، يستخدم دالة فقدان الخطأ المربع، والإخراج هو
مجموعة من القيم المستمرة.

:class:`MLPRegressor` يدعم أيضًا التراجع متعدد الإخراج، حيث
يمكن أن يكون للعينة أكثر من هدف واحد.

التنظيم
==============

كل من: class:`MLPRegressor` و: class:`MLPClassifier` تستخدم المعلمة "alpha"
للتنظيم (L2 التنظيم) الذي يساعد في تجنب الإفراط في التهيئة
من خلال معاقبة الأوزان ذات القيم الكبيرة. يوضح الرسم البياني التالي قرار وظيفة متغيرة مع قيمة alpha.

.. figure:: ../auto_examples/neural_networks/images/sphx_glr_plot_mlp_alpha_001.png
   :target: ../auto_examples/neural_networks/plot_mlp_alpha.html
   :align: center
   :scale: 75

راجع الأمثلة أدناه لمزيد من المعلومات.

.. rubric:: أمثلة

* :ref:`sphx_glr_auto_examples_neural_networks_plot_mlp_alpha.py`

الخوارزميات
==========

يتدرب MLP باستخدام `Stochastic Gradient Descent
<https://en.wikipedia.org/wiki/Stochastic_gradient_descent>`_،
:arxiv:`Adam <1412.6980>`، أو
`L-BFGS <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`__.
Stochastic Gradient Descent (SGD) يقوم بتحديث المعلمات باستخدام تدرج دالة الفقدان فيما يتعلق بمعلمة تحتاج إلى التكيف، أي

.. math::

    w \leftarrow w - \eta (\alpha \frac{\partial R(w)}{\partial w}
    + \frac{\partial Loss}{\partial w})

حيث:math:`\eta` هو معدل التعلم الذي يتحكم في حجم الخطوة
في البحث في مساحة المعلمة.  :math:`Loss` هي دالة الفقدان المستخدمة
للشبكة.

مزيد من التفاصيل يمكن العثور عليها في وثائق
`SGD <https://scikit-learn.org/stable/modules/sgd.html>`_

Adam مشابه لـ SGD من حيث أنه محسن عشوائي، ولكنه يمكن
تعديل مقدار التحديث تلقائيًا بناءً على تقديرات لحظية منخفضة المستوى.

مع SGD أو Adam، يدعم التدريب التعلم عبر الإنترنت والتعلم المصغر.

L-BFGS هو محسن يقرب مصفوفة هيسيان التي تمثل
المشتق الجزئي من الدرجة الثانية لدالة. علاوة على ذلك، فهو يقرب
عكس مصفوفة هيسيان لأداء تحديثات المعلمات. يستخدم التنفيذ
إصدار Scipy من `L-BFGS
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.fmin_l_bfgs_b.html>`_.

إذا كان المحلل المختار هو "L-BFGS"، فإن التدريب لا يدعم التعلم عبر الإنترنت ولا
التعلم المصغر.


التعقيد
==========

افترض أن هناك:math:`n` عينات التدريب،:math:`m` الميزات،:math:`k`
الطبقات المخفية، كل منها يحتوي على:math:`h` العصبونات - للبساطة، و:math:`o`
عصبونات الإخراج.  التعقيد الزمني للتراجع هو
:math:`O(i \cdot n \cdot (m \cdot h + (k - 1) \cdot h \cdot h + h \cdot o))`، حيث:math:`i` هو عدد
الحلقات. نظرًا لأن التراجع له تعقيد زمني مرتفع، فمن المستحسن
بدء التدريب بعدد صغير من العصبونات المخفية وعدد قليل من الطبقات.

.. dropdown:: الصيغة الرياضية

  بالنظر إلى مجموعة من أمثلة التدريب:math:`(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)`
  حيث:math:`x_i \in \mathbf{R}^n` و:math:`y_i \in \{0, 1\}`، فإن طبقة واحدة
  طبقة خفية واحدة MLP تتعلم الدالة:math:`f(x) = W_2 g(W_1^T x + b_1) + b_2`
  حيث:math:`W_1 \in \mathbf{R}^m` و:math:`W_2, b_1, b_2 \in \mathbf{R}` هي
  معلمات النموذج. :math:`W_1, W_2` تمثل أوزان طبقة الإدخال والطبقة المخفية، على التوالي؛ و:math:`b_1, b_2` تمثل التحيز المضافة
  إلى الطبقة المخفية وطبقة الإخراج، على التوالي.
  :math:`g(\cdot) : R \rightarrow R` هي دالة التنشيط، يتم تعيينها افتراضيًا كـ
  دالة تانغينس الزائدية. يتم إعطاؤها على النحو التالي،

  .. math::
        g(z)= \frac{e^z-e^{-z}}{e^z+e^{-z}}

  للتصنيف الثنائي،:math:`f(x)` يمر عبر الدالة اللوغاريتمية
  :math:`g(z)=1/(1+e^{-z})` للحصول على قيم الإخراج بين الصفر والواحد. سيتم تعيين عتبة، محددة على 0.5، لعينات الإخراج الأكبر أو تساوي 0.5
  إلى الفئة الإيجابية، والباقي إلى الفئة السلبية.

  إذا كان هناك أكثر من فئتين، فإن:math:`f(x)` نفسها ستكون متجهًا من
  الحجم (n_classes,). بدلاً من المرور عبر الدالة اللوغاريتمية، فإنه يمر
  عبر دالة softmax، والتي يتم كتابتها على النحو التالي،

  .. math::
        \text{softmax}(z)_i = \frac{\exp(z_i)}{\sum_{l=1}^k\exp(z_l)}

  حيث:math:`z_i` يمثل:math:`i` العنصر من الإدخال إلى softmax،
  والذي يتوافق مع الفئة:math:`i`، و:math:`K` هو عدد الفئات.
  النتيجة هي متجه يحتوي على احتمالات أن العينة:math:`x`
  تنتمي إلى كل فئة. الإخراج هو الفئة ذات الاحتمالية الأعلى.

  في التراجع، يظل الإخراج كما هو:math:`f(x)`؛ لذلك، دالة تنشيط الإخراج
  هي مجرد دالة الهوية.

  يستخدم MLP دالات فقدان مختلفة اعتمادًا على نوع المشكلة. دالة الفقدان للتصنيف هي متوسط Cross-Entropy، والتي في الحالة الثنائية تعطى على النحو التالي،

  .. math::

      Loss(\hat{y},y,W) = -\dfrac{1}{n}\sum_{i=0}^n(y_i \ln {\hat{y_i}} + (1-y_i) \ln{(1-\hat{y_i})}) + \dfrac{\alpha}{2n} ||W||_2^2

  حيث:math:`\alpha ||W||_2^2` هي دالة تنظيم L2 (المعروفة باسم العقوبة)
  التي تعاقب النماذج المعقدة؛ و:math:`\alpha > 0` هو معلمة غير سلبية
  يتحكم في حجم العقوبة.

  للتراجع، يستخدم MLP دالة فقدان متوسط مربع الخطأ، والتي تكتب على النحو التالي،

  .. math::

      Loss(\hat{y},y,W) = \frac{1}{2n}\sum_{i=0}^n||\hat{y}_i - y_i ||_2^2 + \frac{\alpha}{2n} ||W||_2^2

  بدءًا من أوزان عشوائية أولية، يقلل متعدد الطبقات Perceptron (MLP)
  دالة الفقدان عن طريق تحديث هذه الأوزان بشكل متكرر. بعد حساب الفقدان، يقوم تمرير خلفي بنشره من طبقة الإخراج إلى الطبقات السابقة، مما يوفر لكل وزن معلمة بقيمة تحديث تهدف إلى تقليل الفقدان.

  في التدرج النازل، يتم حساب تدرج:math:`\nabla Loss_{W}` لدالة الفقدان فيما يتعلق
  بالأوزان ويتم خصمها من:math:`W`.
  يتم التعبير عنه بشكل أكثر رسمية على النحو التالي،

  .. math::
      W^{i+1} = W^i - \epsilon \nabla {Loss}_{W}^{i}

  حيث:math:`i` هي خطوة الحلقة، و:math:`\epsilon` هو معدل التعلم
  مع قيمة أكبر من 0.

  يتوقف الخوارزمية عند الوصول إلى عدد محدد مسبقًا من الحلقات؛ أو
  عندما يكون التحسن في الفقدان أقل من رقم معين، صغير.


.. _mlp_tips:

نصائح حول الاستخدام العملي
=====================

* متعدد الطبقات Perceptron حساس لقياس الميزة، لذا
  يوصى بشدة بتصنيف بياناتك. على سبيل المثال، قم بتصنيف كل
  ميزة على مصفوفة X إلى [0، 1] أو [-1، +1]، أو قم بتصنيفها
  باستخدام: class:`~sklearn.preprocessing.StandardScaler`

    >>> from sklearn.preprocessing import StandardScaler  # doctest: +SKIP
    >>> scaler = StandardScaler()  # doctest: +SKIP
    >>> # Don't cheat - fit only on training data
    >>> scaler.fit(X_train)  # doctest: +SKIP
    >>> X_train = scaler.transform(X_train)  # doctest: +SKIP
    >>> # apply same transformation to test data
    >>> X_test = scaler.transform(X_test)  # doctest: +SKIP

  يعد النهج البديل والموصى به هو استخدام
  :class:`~sklearn.preprocessing.StandardScaler` في
  :class:`~sklearn.pipeline.Pipeline`

* العثور على معلمة التنظيم المعقولة:math:`\alpha` يتم بشكل أفضل
  باستخدام: class:`~sklearn.model_selection.GridSearchCV`، عادة في النطاق
  ``10.0 ** -np.arange(1, 7)``.

* من الناحية التجريبية، لاحظنا أن `L-BFGS` يتقارب بشكل أسرع ومع
  حلول أفضل لمجموعات البيانات الصغيرة. بالنسبة لمجموعات البيانات الكبيرة نسبيًا، ومع ذلك، فإن `Adam` قوي جدًا. عادة ما يتقارب بسرعة
  ويعطي أداءً جيدًا جدًا. يمكن لـ `SGD` مع الزخم أو
  زخم nesterov، من ناحية أخرى، أن يؤدي أداءً أفضل من
  هذين الخوارزميتين إذا تم ضبط معدل التعلم بشكل صحيح.

مزيد من التحكم مع warm_start
============================
إذا كنت تريد مزيدًا من التحكم في معايير التوقف أو معدل التعلم في SGD،
أو تريد إجراء مراقبة إضافية، يمكن أن يكون استخدام "warm_start=True" و
"max_iter=1" والقيام بالحلقة بنفسك مفيدًا::

    >>> X = [[0., 0.], [1., 1.]]
    >>> y = [0, 1]
    >>> clf = MLPClassifier(hidden_layer_sizes=(15,), random_state=1, max_iter=1, warm_start=True)
    >>> for i in range(10):
    ...     clf.fit(X, y)
    ...     # additional monitoring / inspection
    MLPClassifier(...

.. dropdown:: مراجع

  * `"Learning representations by back-propagating errors."
    <https://www.iro.umontreal.ca/~pift6266/A06/refs/backprop_old.pdf>`_
    Rumelhart, David E., Geoffrey E. Hinton, and Ronald J. Williams.

  * `"Stochastic Gradient Descent" <https://leon.bottou.org/projects/sgd>`_ L. Bottou - Website, 2010.

  * `"Backpropagation" <http://ufldl.stanford.edu/wiki/index.php/Backpropagation_Algorithm>`_
    Andrew Ng, Jiquan Ngiam, Chuan Yu Foo, Yifan Mai, Caroline Suen - Website, 2011.

  * `"Efficient BackProp" <http://yann.lecun.com/exdb/publis/pdf/lecun-98b.pdf>`_
    Y. LeCun, L. Bottou, G. Orr, K. Müller - In Neural Networks: Tricks of the Trade 1998.

  * :arxiv:`"Adam: A method for stochastic optimization." <1412.6980>`
    Kingma, Diederik, and Jimmy Ba (2014)