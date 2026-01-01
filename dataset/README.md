
## 📁 Data Organization
To maintain consistency with the five-fold cross-validation and the 8:2 train-test split used in our study, 
please organize the BUS-BRA dataset  using the following structure.

```
dataset/
└── BUSBRA/
    └── five-fold/
        ├── fold1/
        │   ├── train/
        │   │   ├── images/
        │   │   │   ├── patient_001_A.png
                    ├── patient_002_B.png  
        │   │   │   └── ...
        │   │   └── labels/
        │   │       ├── patient_001_A.png
                    ├── patient_002_B.png 
        │   │       └── ...
        │   └── test/
        │       ├── images/
        │       │   ├── patient_099_C.png
        │       │   └── ...
        │       └── labels/
        │           ├── patient_099_C.png
        │           └── ...
        ├── fold2/
        │   ├── ...
        ├── fold3/
        │   ├── ...
        ├── fold4/
        │   ├── ...
        └── fold5/
            ├── train/
            └── test/
```
