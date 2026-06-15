# Full PCG Murmur Evaluation Report

## Best patient-level result per fold

|   fold |   epoch |   accuracy |   balanced_accuracy |   precision |   recall_sensitivity |   specificity |       f1 |    auroc |   tn |   fp |   fn |   tp | file                                           |
|-------:|--------:|-----------:|--------------------:|------------:|---------------------:|--------------:|---------:|---------:|-----:|-----:|-----:|-----:|:-----------------------------------------------|
|      0 |      11 |   0.942857 |            0.914316 |    0.864865 |             0.864865 |      0.963768 | 0.864865 | 0.96083  |  133 |    5 |    5 |   32 | outputs/predictions/fold0_epoch11_patients.csv |
|      1 |       3 |   0.657143 |            0.781022 |    0.387755 |             1        |      0.562044 | 0.558824 | 0.966577 |   77 |   60 |    0 |   38 | outputs/predictions/fold1_epoch3_patients.csv  |
|      2 |      17 |   0.914286 |            0.85753  |    0.787879 |             0.764706 |      0.950355 | 0.776119 | 0.941176 |  134 |    7 |    8 |   26 | outputs/predictions/fold2_epoch17_patients.csv |
|      3 |      15 |   0.885057 |            0.760123 |    0.75     |             0.5625   |      0.957746 | 0.642857 | 0.8739   |  136 |    6 |   14 |   18 | outputs/predictions/fold3_epoch15_patients.csv |
|      4 |      10 |   0.914286 |            0.869189 |    0.810811 |             0.789474 |      0.948905 | 0.8      | 0.93738  |  130 |    7 |    8 |   30 | outputs/predictions/fold4_epoch10_patients.csv |

## Cross-fold patient-level mean ± std

- **accuracy**: 0.8627 ± 0.1044
- **balanced_accuracy**: 0.8364 ± 0.0574
- **precision**: 0.7203 ± 0.1704
- **recall_sensitivity**: 0.7963 ± 0.1428
- **specificity**: 0.8766 ± 0.1574
- **f1**: 0.7285 ± 0.1114
- **auroc**: 0.9360 ± 0.0330

## Best segment-level result per fold

|   fold |   epoch |   accuracy |   balanced_accuracy |   precision |   recall_sensitivity |   specificity |       f1 |    auroc |   tn |   fp |   fn |   tp | file                                           |
|-------:|--------:|-----------:|--------------------:|------------:|---------------------:|--------------:|---------:|---------:|-----:|-----:|-----:|-----:|:-----------------------------------------------|
|      0 |       3 |   0.771344 |            0.776953 |    0.479343 |             0.786765 |      0.767142 | 0.595732 | 0.873472 | 3446 | 1046 |  261 |  963 | outputs/predictions/fold0_epoch3_segments.csv  |
|      1 |       2 |   0.71846  |            0.759509 |    0.428112 |             0.832812 |      0.686205 | 0.565517 | 0.869067 | 3114 | 1424 |  214 | 1066 | outputs/predictions/fold1_epoch2_segments.csv  |
|      2 |      22 |   0.83327  |            0.738487 |    0.529412 |             0.591102 |      0.885872 | 0.558559 | 0.833651 | 3850 |  496 |  386 |  558 | outputs/predictions/fold2_epoch22_segments.csv |
|      3 |       3 |   0.860244 |            0.708362 |    0.638484 |             0.474026 |      0.942699 | 0.544099 | 0.803672 | 4080 |  248 |  486 |  438 | outputs/predictions/fold3_epoch3_segments.csv  |
|      4 |      10 |   0.857444 |            0.770108 |    0.665146 |             0.621483 |      0.918733 | 0.642574 | 0.858257 | 4149 |  367 |  444 |  729 | outputs/predictions/fold4_epoch10_segments.csv |
## Training history files

- `outputs/logs/history_fold0.json`: epochs=18, last_epoch=18, best_auroc_epoch=11, best_auroc=0.9608303956130042
- `outputs/logs/history_fold1.json`: epochs=12, last_epoch=12, best_auroc_epoch=3, best_auroc=0.9665770265078756
- `outputs/logs/history_fold2.json`: epochs=24, last_epoch=24, best_auroc_epoch=17, best_auroc=0.9411764705882353
- `outputs/logs/history_fold3.json`: epochs=26, last_epoch=26, best_auroc_epoch=15, best_auroc=0.8738996478873241
- `outputs/logs/history_fold4.json`: epochs=10, last_epoch=10, best_auroc_epoch=10, best_auroc=0.9373799462159047
