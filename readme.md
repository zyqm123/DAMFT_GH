This repository contains the data and code for the paper named "Combining data augmentation and model fine-tuning for learning from limited data".

Install
Install dependencies with pip install -r requirements.txt

Datasets
Datasets used in this study are selected from UCI and KEEL dataset repositories, as shown in the "data" directory.

Code
The core code consists of three parts, as shown in the "code/code" directory:

1. Data Generation:
   - Script: 
	-DAMFT_GH_generate.py

2Classification:
   - Script: DAMFT_GH_classifier.py

Execution Commands:
To reproduce the results published in the paper, execute the following commands sequentially:
```
python DAMFT_GH_generate.py
python DAMFT_GH_classifier.py
```