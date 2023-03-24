---
title: "Анализ изображений, МФТИ"
author: [Александр Жуковский]
date: "11.11.2022"
keywords: [miptcv, Metric Learning]
...

# Задача №7. Метрическое обучение

[![PyTorch](https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://pytorchlightning.ai/)
[![Hydra](https://img.shields.io/badge/Config-Hydra-89b8cd)](https://hydra.cc/)
[![Template](https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray)](https://github.com/ashleve/lightning-hydra-template)
[![PyTorch Metric learning](https://img.shields.io/badge/-PyTorch Metric learning-ef5552?logo=github&labelColor=gray)](https://github.com/KevinMusgrave/pytorch-metric-learning)
[![MLflow](https://img.shields.io/badge/-MLfLow-4bc8ea?logo=mlflow&logoColor=white)](https://mlflow.org/)
[![TensorBoard](https://img.shields.io/badge/-TensorBoard-f36f00?logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/tensorboard)
[![TorchMetrics](https://img.shields.io/badge/-TorchMetrics-792ee5?logo=github&labelColor=gray)](https://torchmetrics.readthedocs.io/en/latest/)

В задаче Вам предстоит познакомиться с современными инструментами для проведения экспериментов с машинным обучением, включая обучение, тестирование, сравнение и отслеживание качества их работы. Вам потребуется спроектировать и реализовать механизм вычисления точности классификации предварительно обученной модели метрического обучения.

Работа будет проводиться на наборе данных [```Market-1501```](https://zheng-lab.cecs.anu.edu.au/Project/project_reid.html). Он состоит из изображений людей, снятых с разных камер и разделен на 2 непересекающиеся по людям части: обучающую и тестовую. Набор подготовлен к обучению и избавлен от отвлекающих классов и мусора, его требуется только распаковать в папку data.

На обучающей части (c выделением из неё валидационой) была обучена некоторая нейронная сеть с функцией потерь ArcFace (```models/epoch_120.ckpt```). Требуется определить качество работы этой сети на изображениях людей, которые не входили в обучающую выборку.

Тестовая часть была также разделена на две: эталон (```reference```) -- изображения с известным соответствием людям и запрос (```query```) -- изображения, для которых это соответствие требуется определить. Для определения качества работы сети нужно оценить, насколько точно сеть, зная соответствия эталона, может предсказать соответствия изображениям запроса людям. Для определения соответствия требуется использовать все изображения из эталона, обучающая выборка в процессе тестирования участия не принимает.

Механизм вычисления точности классификатора требуется реализовать непосредственно в коде репозитория. Точность классификации должна логироваться в ```MLflow``` при запуске ```python eval.py```. Можете использовать любые дополнительных библиотеки, хотя предоставленных достаточно для решения. Используемые библиотеки и шаблон репозитория указаны сверху страницы и хорошо документированы. Модифицировать и отправлять (без папок!) можно только файлы ```src/models/metric_learning_module.py``` и ```src/datamodules/market_datamodule.py```.

Для проверки решения можете использовать ```pytest```:
```bash
pytest tests/test_metric_learning.py

```