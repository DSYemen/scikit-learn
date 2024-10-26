.. _naive_bayes:

===========
خوارزميات بايز الساذجة
===========

.. currentmodule:: sklearn.naive_bayes


خوارزميات بايز الساذجة هي مجموعة من خوارزميات التعلم الخاضع للإشراف
تستند إلى تطبيق نظرية بايز "الساذجة" للافتراض "الساذج" للاستقلال الشرطي
بين كل زوج من الميزات بالنظر إلى قيمة متغير الفئة. تنص نظرية بايز على العلاقة التالية،
نظرًا لمتغير الفئة:math:`y` والميزة التابعة المتجه:math:`x_1` من خلال:math:`x_n`،:

.. math::

   P(y \mid x_1, \dots, x_n) = \frac{P(y) P(x_1, \dots, x_n \mid y)}
                                    {P(x_1, \dots, x_n)}

باستخدام افتراض الاستقلال الشرطي الساذج الذي

.. math::

   P(x_i | y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i | y),

لكل:math:`i`، يتم تبسيط هذه العلاقة إلى

.. math::

   P(y \mid x_1, \dots, x_n) = \frac{P(y) \prod_{i=1}^{n} P(x_i \mid y)}
                                    {P(x_1, \dots, x_n)}

نظرًا لأن:math:`P(x_1, \dots, x_n)` ثابت بالنظر إلى الإدخال،
يمكننا استخدام قاعدة التصنيف التالية:

.. math::

   P(y \mid x_1, \dots, x_n) \propto P(y) \prod_{i=1}^{n} P(x_i \mid y)

   \Downarrow

   \hat{y} = \arg\max_y P(y) \prod_{i=1}^{n} P(x_i \mid y),

ويمكننا استخدام تقدير أقصى احتمال لاحق (MAP) لتقدير
:math:`P(y)` و:math:`P(x_i \mid y)`؛
الأخير هو التردد النسبي للفئة:math:`y`
في مجموعة التدريب.

تختلف خوارزميات بايز الساذجة المختلفة بشكل أساسي بالافتراضات التي
يقومون بها فيما يتعلق بتوزيع:math:`P(x_i \mid y)`.

على الرغم من افتراضاتها التي تبدو مفرطة في التبسيط، فقد عملت خوارزميات بايز الساذجة بشكل جيد جدًا في العديد من الحالات الواقعية، وخاصة
تصنيف المستندات وتصفية الرسائل غير المرغوب فيها. تتطلب كمية صغيرة
من بيانات التدريب لتقدير المعلمات اللازمة. (للأسباب النظرية التي تجعل بايز الساذج يعمل بشكل جيد، وعلى أي أنواع البيانات التي يفعلها، راجع
المراجع أدناه.)

يمكن أن تكون خوارزميات بايز الساذجة والمتعلمين سريعة للغاية مقارنة بالأساليب الأكثر تطوراً.
يعني فصل توزيعات الفئة الشرطية أن كل
يمكن تقدير التوزيع كبعد أحادي.
هذا بدوره يساعد على تخفيف المشاكل الناجمة عن لعنة
الأبعاد.

على الجانب الآخر، على الرغم من أن بايز الساذج معروف كمصنف جيد،
إنه معروف بأنه مقدر سيء، لذا لا ينبغي أخذ احتمالات الإخراج من
"predict_proba" على محمل الجد.

.. dropdown:: مراجع

   * H. Zhang (2004). `The optimality of Naive Bayes.
     <https://www.cs.unb.ca/~hzhang/publications/FLAIRS04ZhangH.pdf>`_
     Proc. FLAIRS.

.. _gaussian_naive_bayes:

خوارزمية بايز الساذجة الغاوسية
--------------------

:class:`GaussianNB` تنفذ خوارزمية بايز الساذجة الغاوسية للتصنيف. يفترض أن احتمال الميزات غاوسي:

.. math::

   P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma^2_y}} \exp\left(-\frac{(x_i - \mu_y)^2}{2\sigma^2_y}\right)

يتم تقدير المعلمات:math:`\sigma_y` و:math:`\mu_y`
باستخدام أقصى احتمال.

   >>> from sklearn.datasets import load_iris
   >>> from sklearn.model_selection import train_test_split
   >>> from sklearn.naive_bayes import GaussianNB
   >>> X, y = load_iris(return_X_y=True)
   >>> X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)
   >>> gnb = GaussianNB()
   >>> y_pred = gnb.fit(X_train, y_train).predict(X_test)
   >>> print("Number of mislabeled points out of a total %d points : %d"
   ...       % (X_test.shape[0], (y_test != y_pred).sum()))
   Number of mislabeled points out of a total 75 points : 4

.. _multinomial_naive_bayes:

خوارزمية بايز الساذجة متعددة الحدود
-----------------------

:class:`MultinomialNB` تنفذ خوارزمية بايز الساذجة للبيانات متعددة الحدود، وهي واحدة من متغيرين بايز الساذجة الكلاسيكيين المستخدمين في
تصنيف النصوص (حيث يتم تمثيل البيانات عادةً كمؤشرات متجه الكلمات، على الرغم من أن متجهات tf-idf معروفة أيضًا بالعمل بشكل جيد في الممارسة العملية).
يتم معلمته بواسطة المتجهات
:math:`\theta_y = (\theta_{y1},\ldots,\theta_{yn})`
لكل فئة:math:`y`، حيث:math:`n` هو عدد الميزات
(في تصنيف النصوص، حجم المفردات)
و:math:`\theta_{yi}` هو الاحتمال:math:`P(x_i \mid y)`
لميزة:math:`i` التي تظهر في عينة تنتمي إلى الفئة:math:`y`.

يتم تقدير المعلمات:math:`\theta_y` بواسطة نسخة ملساء
من أقصى احتمال، أي العد النسبي:

.. math::

    \hat{\theta}_{yi} = \frac{ N_{yi} + \alpha}{N_y + \alpha n}

حيث:math:`N_{yi} = \sum_{x \in T} x_i` هو
عدد المرات التي تظهر فيها الميزة:math:`i` في عينة من الفئة:math:`y`
في مجموعة التدريب:math:`T`،
و:math:`N_{y} = \sum_{i=1}^{n} N_{yi}` هو العدد الإجمالي
لجميع الميزات للفئة:math:`y`.

تأخذ معلمات التمهيد:math:`\alpha \ge 0` في الاعتبار
الميزات غير الموجودة في عينات التعلم وتمنع الاحتمالات الصفرية
في الحسابات اللاحقة.
يُطلق على تعيين:math:`\alpha = 1` اسم التمهيد Laplace،
في حين أن:math:`\alpha < 1` يُطلق عليه اسم التمهيد Lidstone.

.. _complement_naive_bayes:

خوارزمية بايز الساذجة التكميلية
----------------------

:class:`ComplementNB` تنفذ خوارزمية بايز الساذجة التكميلية (CNB).
CNB هو تكيف لخوارزمية بايز الساذجة متعددة الحدود (MNB)
التي تناسب بشكل خاص مجموعات البيانات غير المتوازنة. على وجه التحديد، يستخدم CNB
إحصاءات من تكملة كل فئة لحساب أوزان النموذج. يظهر مخترعو CNB تجريبيًا أن تقديرات المعلمات لـ CNB
أكثر استقرارًا من تلك الخاصة بـ MNB. علاوة على ذلك، يتفوق CNB بشكل منتظم على MNB (غالبًا
بهامش كبير) في مهام تصنيف النصوص.

.. dropdown:: حساب الأوزان

   تتمثل الإجراءات لحساب الأوزان فيما يلي:

   .. math::

      \hat{\theta}_{ci} = \frac{\alpha_i + \sum_{j:y_j \neq c} d_{ij}}
                              {\alpha + \sum_{j:y_j \neq c} \sum_{k} d_{kj}}

      w_{ci} = \log \hat{\theta}_{ci}

      w_{ci} = \frac{w_{ci}}{\sum_{j} |w_{cj}|}

   حيث يتم إجراء عمليات الجمع على جميع المستندات:math:`j` غير الموجودة في الفئة:math:`c`،
   :math:`d_{ij}` هو إما عدد أو قيمة tf-idf للمصطلح:math:`i` في المستند
   :math:`j`، :math:`\alpha_i` هو معلمة التمهيد مثل تلك الموجودة في
   MNB، و:math:`\alpha = \sum_{i} \alpha_i`. يعالج التطبيع الثاني
   ميل المستندات الأطول إلى الهيمنة على تقديرات المعلمات في MNB. قاعدة التصنيف هي:

   .. math::

      \hat{c} = \arg\min_c \sum_{i} t_i w_{ci}

   أي، يتم تعيين مستند إلى الفئة التي هي *أفقر* تكملة
   المطابقة.

.. dropdown:: مراجع

   * Rennie, J. D., Shih, L., Teevan, J., & Karger, D. R. (2003).
     `Tackling the poor assumptions of naive bayes text classifiers.
     <https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf>`_
     In ICML (Vol. 3, pp. 616-623).


.. _bernoulli_naive_bayes:

خوارزمية بايز الساذجة متعددة الحدود
---------------------

:class:`BernoulliNB` تنفذ خوارزميات بايز الساذجة للتدريب والتصنيف
لبيانات موزعة وفقًا لتوزيعات متعددة الحدود؛ أي، قد يكون هناك العديد من الميزات ولكن يُفترض أن كل منها
قيمة ثنائية (بيرنولي، منطقية).

تستند قاعدة القرار لبايز الساذج متعدد الحدود إلى

.. math::

    P(x_i \mid y) = P(x_i = 1 \mid y) x_i + (1 - P(x_i = 1 \mid y)) (1 - x_i)

والذي يختلف عن قاعدة بايز الساذج متعددة الحدود
في أنه يعاقب صراحةً على عدم حدوث ميزة:math:`i`
التي هي مؤشر للفئة:math:`y`،
حيث تتجاهل المتغيرات متعددة الحدود غياب ميزة.

في حالة تصنيف النصوص، قد تستخدم متجهات حدوث الكلمات (بدلاً من متجهات عدد الكلمات) لتدريب وتصنيف هذا المصنف. :class:`BernoulliNB`
قد يؤدي أداء أفضل على بعض مجموعات البيانات، خاصة تلك ذات المستندات الأقصر.
من المستحسن تقييم كلا النموذجين، إذا سمح الوقت بذلك.

.. dropdown:: مراجع

   * C.D. Manning, P. Raghavan and H. Schütze (2008). Introduction to
     Information Retrieval. Cambridge University Press, pp. 234-265.

   * A. McCallum and K. Nigam (1998).
     `A comparison of event models for Naive Bayes text classification.
     <https://citeseerx.ist.psu.edu/doc_view/pid/04ce064505b1635583fa0d9cc07cac7e9ea993cc>`_
     Proc. FLAIRS-98 Workshop on Learning for Text Categorization, pp. 41-48.

   * V. Metsis, I. Androutsopoulos and G. Paliouras (2006).
     `Spam filtering with Naive Bayes -- Which Naive Bayes?
     <https://citeseerx.ist.psu.edu/doc_view/pid/8bd0934b366b539ec95e683ae39f8abb29ccc757>`_
     3rd Conf. on Email and Anti-Spam (CEAS).


.. _categorical_naive_bayes:

خوارزمية بايز الساذجة التصنيفية
-----------------------

:class:`CategoricalNB` تنفذ خوارزمية بايز الساذجة
للبيانات التصنيفية. يفترض أن كل ميزة،
التي يتم وصفها بواسطة الفهرس:math:`i`، لها توزيعها التصنيفي الخاص.

بالنسبة لكل ميزة:math:`i` في مجموعة التدريب:math:`X`،
:class:`CategoricalNB` يقدر توزيعًا تصنيفيًا لكل ميزة i
من X مشروطة على الفئة y. يتم تعريف مجموعة فهرس العينات على أنها
:math:`J`، مع:math:`m` كعدد العينات.

.. dropdown:: حساب الاحتمال

   يتم تقدير احتمال الفئة:math:`t` في الميزة:math:`i` بالنظر إلى الفئة
   :math:`c` كما يلي:

   .. math::

      P(x_i = t \mid y = c \: ;\, \alpha) = \frac{ N_{tic} + \alpha}{N_{c} +
                                             \alpha n_i},

   حيث:math:`N_{tic} = |\{j \in J \mid y_j = c\}|` هو عدد
   المرات التي تظهر فيها الفئة:math:`t` في العينات:math:`x_{i}`، والتي تنتمي
   إلى الفئة:math:`c`، :math:`N_{c} = |\{ j \in J\mid y_j = c\}|` هو عدد العينات
   الفئة c، :math:`n_i` هو عدد الفئات المتاحة
   للميزة:math:`i`.

:class:`CategoricalNB` يفترض أن مصفوفة العينة:math:`X` مشفرة (على سبيل المثال بمساعدة:class:`~sklearn.preprocessing.OrdinalEncoder`)
بحيث يتم تمثيل جميع الفئات لكل ميزة:math:`i` بالأرقام
:math:`0, ..., n_i - 1` حيث:math:`n_i` هو عدد الفئات المتاحة
للميزة:math:`i`.

نموذج بايز الساذج خارج النواة
---------------------------------

يمكن استخدام نماذج بايز الساذجة لمعالجة مشاكل التصنيف واسعة النطاق
بالنسبة للتي قد لا تتسع مجموعة التدريب بالكامل في الذاكرة. لمعالجة هذه الحالة،
:class:`MultinomialNB`، و:class:`BernoulliNB`، و:class:`GaussianNB`
تعرض طريقة "partial_fit" التي يمكن استخدامها
بشكل متزايد كما هو الحال مع المصنفات الأخرى كما هو موضح في
:ref:`sphx_glr_auto_examples_applications_plot_out_of_core_classification.py`. جميع مصنفات بايز الساذجة
تدعم وزن العينة.

على عكس طريقة "fit"، تحتاج المكالمة الأولى لـ "partial_fit" إلى
تمرير قائمة بجميع التصنيفات المتوقعة.

للاطلاع على نظرة عامة على الاستراتيجيات المتاحة في scikit-learn، راجع أيضًا
:ref:`out-of-core learning <scaling_strategies>` الوثائق.

.. note::

   تُدخل طريقة "partial_fit" لنموذج بايز الساذج بعض
   النفقات العامة الحسابية. يوصى باستخدام أحجام بيانات الشرائح التي تكون كبيرة قدر الإمكان، أي كما يسمح بها RAM المتاح.