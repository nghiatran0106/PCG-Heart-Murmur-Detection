# Patient-level Evaluation, Best Epoch Selected by F1

| Fold    |   Accuracy |   Precision |   Recall/Sensitivity |   Specificity |   F1-Score |   AUC-ROC |   Average Precision |   Balanced Accuracy |    MCC |   Cohen Kappa |
|:--------|-----------:|------------:|---------------------:|--------------:|-----------:|----------:|--------------------:|--------------------:|-------:|--------------:|
| Fold 0  |     0.9429 |      0.8649 |               0.8649 |        0.9638 |     0.8649 |    0.9608 |              0.9332 |              0.9143 | 0.8286 |        0.8286 |
| Fold 1  |     0.9314 |      0.8611 |               0.8158 |        0.9635 |     0.8378 |    0.9347 |              0.8794 |              0.8896 | 0.7949 |        0.7944 |
| Fold 2  |     0.9143 |      0.7879 |               0.7647 |        0.9504 |     0.7761 |    0.9412 |              0.8685 |              0.8575 | 0.7233 |        0.7231 |
| Fold 3  |     0.9138 |      0.7742 |               0.75   |        0.9507 |     0.7619 |    0.8506 |              0.7954 |              0.8504 | 0.7094 |        0.7093 |
| Fold 4  |     0.9371 |      0.9355 |               0.7632 |        0.9854 |     0.8406 |    0.9109 |              0.8719 |              0.8743 | 0.8084 |        0.8019 |
| AVERAGE |     0.9279 |      0.8447 |               0.7917 |        0.9627 |     0.8163 |    0.9196 |              0.8697 |              0.8772 | 0.7729 |        0.7715 |

Note: Recall/Sensitivity is computed for the murmur Present class. AUC-ROC, Average Precision, MCC, and Cohen's Kappa are reported from patient-level predictions.
