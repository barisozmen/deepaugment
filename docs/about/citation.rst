Citation & References
=====================

How to cite DeepAugment and related work.

Citing DeepAugment
------------------

If you use DeepAugment in your research, please cite:

BibTeX
~~~~~~

.. code-block:: bibtex

   @software{ozmen2019deepaugment,
     author = {Özmen, Barış},
     title = {DeepAugment: Automated Data Augmentation},
     year = {2019},
     url = {https://github.com/barisozmen/deepaugment},
     doi = {10.5281/zenodo.2949929}
   }

DOI
~~~

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2949929.svg
   :target: https://doi.org/10.5281/zenodo.2949929
   :alt: DOI

Resources
~~~~~~~~~

- **GitHub**: https://github.com/barisozmen/deepaugment
- **PyPI**: https://pypi.org/project/deepaugment/
- **Blog post**: `AutoML for Data Augmentation <https://medium.com/insight-data/automl-for-data-augmentation-e87cf692c366>`_
- **Slides**: `Presentation <https://docs.google.com/presentation/d/1toRUTT9X26ACngr6DXCKmPravyqmaGjy-eIU5cTbG1A/edit#slide=id.g4cc092dbc6_0_0>`_

----

Related Work
------------

AutoAugment
~~~~~~~~~~~

The original work on learned augmentation policies:

.. code-block:: bibtex

   @inproceedings{cubuk2018autoaugment,
     title={AutoAugment: Learning Augmentation Policies from Data},
     author={Cubuk, Ekin D and Zoph, Barret and Mane, Deven and Vasudevan, Vijay and Le, Quoc V},
     booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
     pages={113--123},
     year={2019}
   }

- **Paper**: https://arxiv.org/abs/1805.09501
- **Key idea**: Use Reinforcement Learning to discover augmentation policies
- **DeepAugment difference**: Uses Bayesian Optimization instead of RL (~100x faster)

Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~

The optimization method used by DeepAugment:

.. code-block:: bibtex

   @article{shahriari2016taking,
     title={Taking the human out of the loop: A review of Bayesian optimization},
     author={Shahriari, Bobak and Swersky, Kevin and Wang, Ziyu and Adams, Ryan P and De Freitas, Nando},
     journal={Proceedings of the IEEE},
     volume={104},
     number={1},
     pages={148--175},
     year={2016},
     publisher={IEEE}
   }

- **Paper**: https://ieeexplore.ieee.org/document/7352306
- **Key concept**: Efficiently optimize expensive black-box functions
- **Application in DeepAugment**: Search augmentation policy space

Neural Architecture Search
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Related methodology for neural network design:

.. code-block:: bibtex

   @inproceedings{zoph2016neural,
     title={Neural architecture search with reinforcement learning},
     author={Zoph, Barret and Le, Quoc V},
     booktitle={International Conference on Learning Representations},
     year={2017}
   }

- **Paper**: https://arxiv.org/abs/1611.01578
- **Relation**: Similar search methodology, different problem domain

Cutout
~~~~~~

Random occlusion augmentation technique:

.. code-block:: bibtex

   @inproceedings{devries2017improved,
     title={Improved regularization of convolutional neural networks with cutout},
     author={DeVries, Terrance and Taylor, Graham W},
     booktitle={arXiv preprint arXiv:1708.04552},
     year={2017}
   }

- **Paper**: https://arxiv.org/abs/1708.04552
- **Included in DeepAugment**: As one of the 26 available transforms

----

Blog Posts & Tutorials
----------------------

Understanding Bayesian Optimization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- `A Conceptual Explanation of Bayesian Optimization <https://towardsdatascience.com/a-conceptual-explanation-of-bayesian-model-based-hyperparameter-optimization-for-machine-learning-b8172278050f>`_ (Towards Data Science)
- `Bayesian Optimization Primer <https://app.sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf>`_ (SigOpt whitepaper)
- `Let's Talk Bayesian Optimization <https://mlconf.com/lets-talk-bayesian-optimization/>`_ (MLconf)

Data Augmentation
~~~~~~~~~~~~~~~~~

- `AutoML for Data Augmentation <https://medium.com/insight-data/automl-for-data-augmentation-e87cf692c366>`_ (DeepAugment blog post)
- `The Effectiveness of Data Augmentation in Image Classification <https://arxiv.org/abs/1712.04621>`_
- `Bag of Tricks for Image Classification <https://arxiv.org/abs/1812.01187>`_

----

Dependencies
------------

DeepAugment builds on these excellent open-source projects:

Core Libraries
~~~~~~~~~~~~~~

**PyTorch**

.. code-block:: bibtex

   @incollection{pytorch2019,
     title = {PyTorch: An Imperative Style, High-Performance Deep Learning Library},
     author = {Paszke, Adam and Gross, Sam and Massa, Francisco and Lerer, Adam and Bradbury, James and Chanan, Gregory and Killeen, Trevor and Lin, Zeming and Gimelshein, Natalia and Antiga, Luca and Desmaison, Alban and Kopf, Andreas and Yang, Edward and DeVito, Zachary and Raison, Martin and Tejani, Alykhan and Chilamkurthy, Sasank and Steiner, Benoit and Fang, Lu and Bai, Junjie and Chintala, Soumith},
     booktitle = {Advances in Neural Information Processing Systems 32},
     pages = {8024--8035},
     year = {2019},
     publisher = {Curran Associates, Inc.}
   }

**scikit-optimize**

.. code-block:: bibtex

   @misc{skopt,
     author = {Tim Head and MechCoder and Gilles Louppe and Iaroslav Shcherbatyi and fcharras and Zé Vinícius and cmmalone and Christopher Schröder and nel215 and Nuno Campos and Todd Young and Stefano Cereda and Thomas Fan and rene-rex and Kejia (KJ) Shi and Justus Schwabedal and carlosdanielcsantos and Hvass-Labs and Mikhail Pak and SoManyUsernamesTaken and Fred Callaway and Loïc Estève and Lilian Besson and Mehdi Cherti and Karlson Pfannschmidt and Fabian Linzberger and Christophe Cauet and Anna Gut and Andreas Mueller and Alexander Fabisch},
     title = {scikit-optimize/scikit-optimize},
     year = {2018},
     publisher = {Zenodo},
     doi = {10.5281/zenodo.1207017},
     url = {https://doi.org/10.5281/zenodo.1207017}
   }

Other Dependencies
~~~~~~~~~~~~~~~~~~

- **NumPy**: Numerical computing
- **torchvision**: Image transformations and datasets
- **tqdm**: Progress bars
- **matplotlib**: Visualization
- **attrs**: Clean class definitions

See [pyproject.toml](../../pyproject.toml) for complete dependency list.

----

License
-------

DeepAugment is released under the MIT License:

.. code-block:: text

   MIT License

   Copyright (c) 2019-2025 Barış Özmen

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in all
   copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
   SOFTWARE.

----

Contributing
------------

Contributions are welcome! See the `contributing guide <https://github.com/barisozmen/deepaugment/blob/master/CONTRIBUTING.md>`_ for details.

Issues and Questions
~~~~~~~~~~~~~~~~~~~~

- **Bug reports**: `GitHub Issues <https://github.com/barisozmen/deepaugment/issues>`_
- **Feature requests**: `GitHub Discussions <https://github.com/barisozmen/deepaugment/discussions>`_
- **Questions**: `Stack Overflow <https://stackoverflow.com/questions/tagged/deepaugment>`_ (tag: ``deepaugment``)

----

Acknowledgments
---------------

DeepAugment was developed as part of the Insight Data Science program. Special thanks to:

- The PyTorch and torchvision teams for the excellent deep learning framework
- The scikit-optimize team for the Bayesian optimization library
- The AutoAugment authors for pioneering work on learned augmentation
- All contributors and users of DeepAugment

----

See Also
--------

- :doc:`how-it-works` - Technical details
- :doc:`../user-guide/basic-usage` - Getting started
- :doc:`../examples/cifar10` - Example usage
