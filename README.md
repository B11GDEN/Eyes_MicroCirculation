# Eyes_MicroCirculation

## Установка
* Установить pytorch c https://pytorch.org/
* pip install -r requirements.txt

## Датасет
Все датасеты скачайте в папку datasets
* папку train - исходные данные для обучения
* папку test - данные для инференса
* папку EYES_MICRO_AUX - папка была получена путем разметки моделью плохих картинок, обучаться рекомендуется на ней. Ссылка (https://drive.google.com/drive/folders/1zvpCQsnPW9e_RJPXmKPaXFQKX_B0J_Ve?usp=sharing)

После скачивания, чтобы преобразовать данные train в нужный модели формат
```commandline
cd datasets
python3 prepare_data.py
```

## Тренировка
В репозитории представлены обычная (1fold) и ансамблиевая (kfold) тренировка модели.
Для каждого случая написан train.py, inference.py и postprocess.py

Для того, чтобы запустить тренировку на исходных данных
```commandline
cd kfold
python3 train.py -src ../datasets/EYES_MICRO_CLEAR
```
Рекомендуется тренировать на предобработанных данных
```commandline
python3 train.py -src ../datasets/EYES_MICRO_AUX
```
**Важно** Если вы испульзуете Linux то поставьте в train.py флаг -workers равным кол-ву ядер вашего cpu.
В windows -workers 0.

Полученные веса будут находиться в папке *fold/lightning_logs

## Инференс
Для запуска инференса одной модели нужно укfзать прямой путь до весов *.ckpt файла. Пример:
```
cd 1fold
python3 inference.py -weight lightning_logs/checkpoints/epoch=49-val_dice=0.606.ckpt -exp test_inference
```

Для запуска инференса ансамбля нужно указать путь до общей папки. Пример:
```commandline
cd kfold
python3 inference.py -weight lightning_logs -exp test_inference
```
Результаты будут в папке *fold/result/test_inference

## Постобработка
скрипт postprocess.py реализует удаление регионов маленькой площади
```commandline
cd *fold
python3 postprocess.py -src ./result/test_inference -thr 200
```
Удаляет все регионы с площадью < 200 пикселей

Итоговый результат сохранится в папку result/test_inference_post_200

## Рекомендуемая последовательность команд для получения скора на лидерборде

Цифры не устойчивы и могут колебаться в зависимости от экспериментов.
Итоговый мой скор был получен следующим образом

```commandline
cd kfold
python3 train.py -src ../datasets/EYES_MICRO_AUX -num_folds 5 -bs 4 -epochs 200 -workers 8
python3 inference.py -ens_path lightning_logs -thr 0.6 -exp test_inference
python3 postprocess.py -src ./result/test_inference -thr 200
```
