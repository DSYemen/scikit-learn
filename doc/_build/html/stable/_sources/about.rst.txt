.. _about:

حولنا
========

التاريخ
-------

بدأ هذا المشروع في عام 2007 كمشروع Google Summer of Code بواسطة
ديفيد كورنابو. في وقت لاحق من ذلك العام، بدأ ماثيو بروشر العمل على
هذا المشروع كجزء من أطروحته.

في عام 2010، تولى فابيان بيدريجوسا وجايل فاروكس وألكسندر جرامفورت وفينسينت
ميشيل من INRIA قيادة المشروع وأصدروا أول إصدار عام، في الأول من فبراير 2010.
منذ ذلك الحين، ظهرت العديد من الإصدارات وفقًا لدورة مدتها 3 أشهر تقريبًا،
وقاد مجتمع دولي مزدهر عملية التطوير. نتيجة لذلك، تمتلك INRIA حقوق
الطبع والنشر على العمل الذي قام به الأشخاص الذين كانوا يعملون في INRIA في
وقت المساهمة.

الحوكمة
----------

تم تحديد عملية صنع القرار وهيكل حوكمة Scikit-learn في
:ref:`وثيقة الحوكمة <governance>`.

.. توجد مراسي "المؤلف" أدناه لضمان عمل الروابط html القديمة (في
   شكل "about.html#author" لا يزال يعمل)

.. _authors:

الأشخاص وراء Scikit-learn
------------------------------

Scikit-learn هو مشروع مجتمعي، تم تطويره بواسطة مجموعة كبيرة من
الأشخاص، في جميع أنحاء العالم. تلعب بعض الفرق، المدرجة أدناه، أدوارًا
مركزية، ومع ذلك يمكن العثور على قائمة أكثر اكتمالاً بالمساهمين `على
github
<https://github.com/scikit-learn/scikit-learn/graphs/contributors>`__.

فريق الصيانة
................

الأشخاص التالي ذكرهم هم حاليًا مسؤولون عن الصيانة، المسؤولون عن
توحيد تطوير Scikit-learn وصيانته:

.. include:: maintainers.rst

.. note::

  يرجى عدم إرسال بريد إلكتروني إلى المؤلفين مباشرة لطلب المساعدة أو الإبلاغ عن المشكلات.
  بدلاً من ذلك، يرجى مراجعة `ما هي أفضل طريقة لطرح أسئلة حول استخدام Scikit-learn
  <https://scikit-learn.org/stable/faq.html#what-s-the-best-way-to-get-help-on-scikit-learn-usage>`_
  في الأسئلة الشائعة.

.. seealso::

  كيف يمكنك :ref:`المساهمة في المشروع <contributing>`.

فريق التوثيق
..................

يساعد الأشخاص التالي ذكرهم في توثيق المشروع:

.. include:: documentation_team.rst

فريق تجربة المساهمين
...........................

الأشخاص التالي ذكرهم هم مساهمون نشطون يساعدون أيضًا في
:ref:`تصنيف المشكلات <bug_triaging>`_ وطلبات السحب والصيانة العامة:

.. include:: contributor_experience_team.rst

فريق التواصل
..................

يساعد الأشخاص التالي ذكرهم في :ref:`التواصل حول Scikit-learn
<communication_team>`.

.. include:: communication_team.rst

مطورو النواة الفخريون
........................

كان الأشخاص التالي ذكرهم مساهمين نشطين في الماضي، لكنهم لم يعودوا نشطين
في المشروع:

.. include:: maintainers_emeritus.rst

فريق التواصل الفخري
...........................

كان الأشخاص التالي ذكرهم نشطين في فريق التواصل في الماضي، لكنهم لم يعودوا
يتحملون مسؤوليات التواصل:

.. include:: communication_team_emeritus.rst

فريق تجربة المساهمين الفخري
....................................

كان الأشخاص التالي ذكرهم نشطين في فريق تجربة المساهمين في الماضي:

.. include:: contributor_experience_team_emeritus.rst

.. _citing-scikit-learn:

الاستشهاد بـ Scikit-learn
-------------------

إذا كنت تستخدم Scikit-learn في منشور علمي، فنحن نقدر
الاستشهادات بالورقة التالية:

`Scikit-learn: Machine Learning in Python
<https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html>`_, Pedregosa
*et al.*, JMLR 12, pp. 2825-2830, 2011.

إدخال Bibtex::

  @article{scikit-learn,
    title={Scikit-learn: Machine Learning in {P}ython},
    author={Pedregosa, F. and Varoquaux, G. and Gramfort, A. and Michel, V.
            and Thirion, B. and Grisel, O. and Blondel, M. and Prettenhofer, P.
            and Weiss, R. and Dubourg, V. and Vanderplas, J. and Passos, A. and
            Cournapeau, D. and Brucher, M. and Perrot, M. and Duchesnay, E.},
    journal={Journal of Machine Learning Research},
    volume={12},
    pages={2825--2830},
    year={2011}
  }

إذا كنت تريد الاستشهاد بـ Scikit-learn لواجهة برمجة التطبيقات أو تصميمها، فقد ترغب أيضًا في
النظر في الورقة التالية:

:arxiv:`API design for machine learning software: experiences from the scikit-learn
project <1309.0238>`, Buitinck *et al.*, 2013.

إدخال Bibtex::

  @inproceedings{sklearn_api,
    author    = {Lars Buitinck and Gilles Louppe and Mathieu Blondel and
                  Fabian Pedregosa and Andreas Mueller and Olivier Grisel and
                  Vlad Niculae and Peter Prettenhofer and Alexandre Gramfort
                  and Jaques Grobler and Robert Layton and Jake VanderPlas and
                  Arnaud Joly and Brian Holt and Ga{\"{e}}l Varoquaux},
    title     = {{API} design for machine learning software: experiences from the scikit-learn
                  project},
    booktitle = {ECML PKDD Workshop: Languages for Data Mining and Machine Learning},
    year      = {2013},
    pages = {108--122},
  }

الأعمال الفنية
-------

تتوفر شعارات PNG و SVG عالية الجودة في `doc/logos/
<https://github.com/scikit-learn/scikit-learn/tree/main/doc/logos>`_
دليل المصدر.

.. image:: images/scikit-learn-logo-notext.png
  :align: center

التمويل
-------

Scikit-learn هو مشروع مدفوع من قبل المجتمع، ومع ذلك فإن المنح المؤسسية
والخاصة تساعد على ضمان استدامته.

يود المشروع أن يشكر الممولين التالي ذكرهم.

...................................

.. div:: sk-text-image-grid-small

  .. div:: text-box

    توظف `:probabl. <https://probabl.ai>`_ أدريان جلالي وأرتورو آمور
    وفرانسوا جوبيل وجيوم ليمايتر وجيريمي دو بواسبيرانجر ولويك إستيف
    وأوليفييه جريسل وستيفاني سينجر.

  .. div:: image-box

    .. image:: images/probabl.png
      :target: https://probabl.ai

..........

.. |chanel| image:: images/chanel.png
  :target: https://www.chanel.com

.. |axa| image:: images/axa.png
  :target: https://www.axa.fr/

.. |bnp| image:: images/bnp.png
  :target: https://www.bnpparibascardif.com/

.. |dataiku| image:: images/dataiku.png
  :target: https://www.dataiku.com/

.. |nvidia| image:: images/nvidia.png
  :target: https://www.nvidia.com

.. |inria| image:: images/inria-logo.jpg
  :target: https://www.inria.fr

.. raw:: html

  <style>
    table.image-subtable tr {
      border-color: transparent;
    }

    table.image-subtable td {
      width: 50%;
      vertical-align: middle;
      text-align: center;
    }

    table.image-subtable td img {
      max-height: 40px !important;
      max-width: 90% !important;
    }
  </style>

.. div:: sk-text-image-grid-small

  .. div:: text-box

    يساعد `الأعضاء <https://scikit-learn.fondation-inria.fr/en/home/#sponsors>`_
    في `ائتلاف Scikit-learn في مؤسسة Inria
    <https://scikit-learn.fondation-inria.fr/en/home/>`_ في الحفاظ على
    المشروع وتحسينه من خلال دعمهم المالي.

  .. div:: image-box

    .. table::
      :class: image-subtable

      +----------+-----------+
      |       |chanel|       |
      +----------+-----------+
      |  |axa|   |    |bnp|  |
      +----------+-----------+
      |       |nvidia|       |
      +----------+-----------+
      |       |dataiku|      |
      +----------+-----------+
      |        |inria|       |
      +----------+-----------+

..........

.. div:: sk-text-image-grid-small

  .. div:: text-box

    تمول `NVidia <https://nvidia.com>`_ تيم هيد منذ عام 2022
    وهي جزء من ائتلاف Scikit-learn في Inria.

  .. div:: image-box

    .. image:: images/nvidia.png
      :target: https://nvidia.com

..........

.. div:: sk-text-image-grid-small

  .. div:: text-box

    تمول `Microsoft <https://microsoft.com/>`_ أندرياس مولر منذ عام 2020.

  .. div:: image-box

    .. image:: images/microsoft.png
      :target: https://microsoft.com

...........

.. div:: sk-text-image-grid-small

  .. div:: text-box

    تمول `Quansight Labs <https://labs.quansight.org>`_ لوسي ليو منذ عام 2022.

  .. div:: image-box

    .. image:: images/quansight-labs.png
      :target: https://labs.quansight.org

...........

.. |czi| image:: images/czi.png
  :target: https://chanzuckerberg.com

.. |wellcome| image:: images/wellcome-trust.png
  :target: https://wellcome.org/

.. div:: sk-text-image-grid-small

  .. div:: text-box

    تمول `مبادرة تشان زوكربيرج <https://chanzuckerberg.com/>`_ و
    `ويلكوم ترست <https://wellcome.org/>`_ Scikit-learn من خلال
    `برنامج البرامج مفتوحة المصدر الأساسية للعلوم (EOSS) <https://chanzuckerberg.com/eoss/>`_
    الدورة 6.

    وهي تدعم لوسي ليو ومبادرات التنوع والشمول التي سيتم الإعلان
    عنها في المستقبل.

  .. div:: image-box

    .. table::
      :class: image-subtable

      +----------+----------------+
      |  |czi|   |    |wellcome|  |
      +----------+----------------+

...........

.. div:: sk-text-image-grid-small

  .. div:: text-box

    يدعم `Tidelift <https://tidelift.com/>`_ المشروع من خلال اتفاقية الخدمة
    الخاصة بهم.

  .. div:: image-box

    .. image:: images/Tidelift-logo-on-light.svg
      :target: https://tidelift.com/

...........


الرعاة السابقون
.............

.. div:: sk-text-image-grid-small

  .. div:: text-box

    مولت `Quansight Labs <https://labs.quansight.org>`_ ميكيل زين في عامي 2022
    و 2023، ومولت توماس جي فان من عام 2021 إلى عام 2023.

  .. div:: image-box

    .. image:: images/quansight-labs.png
      :target: https://labs.quansight.org

...........

.. div:: sk-text-image-grid-small

  .. div:: text-box

    مولت `جامعة كولومبيا <https://columbia.edu/>`_ أندرياس مولر
    (2016-2020).

  .. div:: image-box

    .. image:: images/columbia.png
      :target: https://columbia.edu

........

.. div:: sk-text-image-grid-small

  .. div:: text-box

    مولت `جامعة سيدني <https://sydney.edu.au/>`_ جويل نوثمان
    (2017-2021).

  .. div:: image-box

    .. image:: images/sydney-primary.jpeg
      :target: https://sydney.edu.au/

...........

.. div:: sk-text-image-grid-small

  .. div:: text-box

    حصل أندرياس مولر على منحة لتحسين Scikit-learn من
    `مؤسسة ألفريد ب. سلون <https://sloan.org>`_ .
    دعمت هذه المنحة منصب نيكولا هاج وتوماس جي فان.

  .. div:: image-box

    .. image:: images/sloan_banner.png
      :target: https://sloan.org/

.............

.. div:: sk-text-image-grid-small

  .. div:: text-box

    تدعم `INRIA <https://www.inria.fr>`_ هذا المشروع بنشاط. لقد قدمت
    تمويلًا لفابيان بيدريجوسا (2010-2012) وجاك جروبلر
    (2012-2013) وأوليفييه جريسل (2013-2017) للعمل على هذا المشروع
    بدوام كامل. كما تستضيف سباقات الترميز والأحداث الأخرى.

  .. div:: image-box

    .. image:: images/inria-logo.jpg
      :target: https://www.inria.fr

.....................

.. div:: sk-text-image-grid-small

  .. div:: text-box

    مول `مركز باريس ساكلاي لعلوم البيانات <http://www.datascience-paris-saclay.fr/>`_
    عامًا واحدًا لمطور للعمل على المشروع بدوام كامل (2014-2015) ، 50٪
    من وقت جيوم ليمايتر (2016-2017) و 50٪ من وقت جوريس فان دين
    بوش (2017-2018).

  .. div:: image-box

    .. image:: images/cds-logo.png
      :target: http://www.datascience-paris-saclay.fr/

..........................

.. div:: sk-text-image-grid-small

  .. div:: text-box

    مولت `بيئة مور سلون لعلوم البيانات بجامعة نيويورك <https://cds.nyu.edu/mooresloan/>`_
    أندرياس مولر (2014-2016) للعمل على هذا المشروع. كما تمول بيئة
    مور سلون لعلوم البيانات العديد من الطلاب للعمل على المشروع
    بدوام جزئي.

  .. div:: image-box

    .. image:: images/nyu_short_color.png
      :target: https://cds.nyu.edu/mooresloan/

........................

.. div:: sk-text-image-grid-small

  .. div:: text-box

    مولت `Télécom Paristech <https://www.telecom-paristech.fr/>`_ مانوج كومار
    (2014) وتوم دوبري لا تور (2015) وراغاف RV (2015-2017) وتيري جيوموت
    (2016-2017) وألبرت توماس (2017) للعمل على Scikit-learn.

  .. div:: image-box

    .. image:: images/telecom.png
      :target: https://www.telecom-paristech.fr/

.....................

.. div:: sk-text-image-grid-small

  .. div:: text-box

    مول `Labex DigiCosme <https://digicosme.lri.fr>`_ نيكولا جويكس
    (2015-2016) وتوم دوبري لا تور (2015-2016 و 2017-2018) وماتورين ماسياس
    (2018-2019) للعمل بدوام جزئي على Scikit-learn خلال دراستهم للدكتوراه.
    كما مول سباق ترميز Scikit-learn في عام 2015.

  .. div:: image-box

    .. image:: images/digicosme.png
      :target: https://digicosme.lri.fr

.....................

.. div:: sk-text-image-grid-small

  .. div:: text-box

    مولت `مبادرة تشان زوكربيرج <https://chanzuckerberg.com/>`_ نيكولا
    هاج للعمل بدوام كامل على Scikit-learn في عام 2020.

  .. div:: image-box

    .. image:: images/czi.png
      :target: https://chanzuckerberg.com

......................

تم رعاية الطلاب التالي ذكرهم من قبل `Google
<https://opensource.google/>`_ للعمل على Scikit-learn من خلال
برنامج `Google Summer of Code <https://en.wikipedia.org/wiki/Google_Summer_of_Code>`_
.

- 2007 - ديفيد كورنابو
- 2011 - `Vlad Niculae`_
- 2012 - `Vlad Niculae`_ ، إيمانويل باير
- 2013 - كمال إرين ، نيكولا تريسينجي
- 2014 - حمزة الصالحي ، عصام لارادجي ، ماهيشاكيا ويجيواردينا ، مانوج كومار
- 2015 - `Raghav RV <https://github.com/raghavrv>`_ ، وي شوي
- 2016 - `Nelson Liu <http://nelsonliu.me>`_ ، `YenChen Lin <https://yenchenlin.me/>`_

.. _Vlad Niculae: https://vene.ro/

...................

يدعم مشروع `NeuroDebian <http://neuro.debian.net>`_ الذي يوفر حزم
`Debian <https://www.debian.org/>`_ والمساهمات من قبل
`Dr. James V. Haxby <http://haxbylab.dartmouth.edu/>`_ (`Dartmouth
College <https://pbs.dartmouth.edu/>`_).

...................

مولت المنظمات التالية ائتلاف Scikit-learn في Inria في الماضي:

.. |msn| image:: images/microsoft.png
  :target: https://www.microsoft.com/

.. |bcg| image:: images/bcg.png
  :target: https://www.bcg.com/beyond-consulting/bcg-gamma/default.aspx

.. |fujitsu| image:: images/fujitsu.png
  :target: https://www.fujitsu.com/global/

.. |aphp| image:: images/logo_APHP_text.png
  :target: https://aphp.fr/

.. |hf| image:: images/huggingface_logo-noborder.png
  :target: https://huggingface.co

.. raw:: html

  <style>
    div.image-subgrid img {
      max-height: 50px;
      max-width: 90%;
    }
  </style>

.. grid:: 2 2 4 4
  :class-row: image-subgrid
  :gutter: 1

  .. grid-item::
    :class: sd-text-center
    :child-align: center

    |msn|

  .. grid-item::
    :class: sd-text-center
    :child-align: center

    |bcg|

  .. grid-item::
    :class: sd-text-center
    :child-align: center

    |fujitsu|

  .. grid-item::
    :class: sd-text-center
    :child-align: center

    |aphp|

  .. grid-item::
    :class: sd-text-center
    :child-align: center

    |hf|

سباقات الترميز
--------------

يمتلك مشروع Scikit-learn تاريخًا طويلًا من `سباقات الترميز مفتوحة المصدر
<https://blog.scikit-learn.org/events/sprints-value/>`_ مع أكثر من 50
حدث سباق من عام 2010 حتى يومنا هذا. هناك العشرات من الرعاة الذين ساهموا
في التكاليف التي تشمل المكان والطعام والسفر ووقت المطور والمزيد. انظر
`سباقات Scikit-learn <https://blog.scikit-learn.org/sprints/>`_ للحصول على قائمة
كاملة بالأحداث.

التبرع للمشروع
-----------------------

إذا كنت مهتمًا بالتبرع للمشروع أو لأحد سباقات الترميز الخاصة بنا،
يرجى التبرع عبر `صفحة تبرعات NumFOCUS
<https://numfocus.org/donate-to-scikit-learn>`_.

.. raw:: html

  <p class="text-center">
    <a class="btn sk-btn-orange mb-1" href="https://numfocus.org/donate-to-scikit-learn">
      ساعدنا، <strong>تبرع!</strong>
    </a>
  </p>

سيتم التعامل مع جميع التبرعات من قبل `NumFOCUS <https://numfocus.org/>`_، وهي منظمة
غير ربحية يديرها مجلس إدارة من `أعضاء مجتمع Scipy
<https://numfocus.org/board.html>`_. تتمثل مهمة NumFOCUS في تعزيز
برمجيات الحوسبة العلمية، لا سيما في Python. بصفتها موطنًا ماليًا لـ Scikit-learn،
فإنها تضمن توفر الأموال عند الحاجة للحفاظ على تمويل المشروع وتوافره
مع الامتثال للوائح الضريبية.

ستخصص التبرعات التي تم تلقيها لمشروع Scikit-learn في الغالب لتغطية
نفقات السفر لسباقات الترميز، بالإضافة إلى ميزانية تنظيم المشروع
[#f1]_.

.. rubric:: ملاحظات

.. [#f1] فيما يتعلق بميزانية التنظيم، على وجه الخصوص، قد نستخدم بعضًا من
  الأموال المتبرع بها لدفع نفقات المشروع الأخرى مثل DNS أو
  خدمات الاستضافة أو التكامل المستمر.


دعم البنية التحتية
----------------------

نود أيضًا أن نشكر `Microsoft Azure <https://azure.microsoft.com/en-us/>`_ و
`Cirrus Cl <https://cirrus-ci.org>`_ و `CircleCl <https://circleci.com/>`_ على وقت
وحدة المعالجة المركزية المجانية على خوادم التكامل المستمر الخاصة بهم، و `Anaconda Inc.
<https://www.anaconda.com>`_ على التخزين الذي يوفرونه لعمليات البناء المرحلية
والليلية الخاصة بنا.