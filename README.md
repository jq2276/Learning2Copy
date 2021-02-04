# README #
----
Codes for AAAI2021 paper ''Learning to Copy Coherent Knowledge for Response Generation.''

## Requirements ##

* tqdm==4.54.0
* numpy==1.19.4
* nltk==3.5
* torch==1.7.1

## Datasets ##
We use two datasets to implement our experiment, one is DuConv and the other is DuRecDial. Put the data under ``data/resource/<DuConv/DuRecDial>`` folder and rename them train/dev/test.txt. (e.g. /data/resource/DuConv/test.txt). The data can be downloaded through the following papers:

**DuConv**: 
> Wu, W.; Guo, Z.; Zhou, X.; Wu, H.; Zhang, X.; Lian, R.;
> and Wang, H. 2019. Proactive Human-Machine Conversation with Explicit Conversation Goal. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 3794–3804.

**DuRecDial**:
> Liu, Z.; Wang, H.; Niu, Z.-Y.; Wu, H.; Che, W.; and Liu, T. 2020. Towards Conversational Recommendation over Multi-Type Dialogs. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 1036–1049. Online: Association for Computational Linguistics.

## Training & Testing ##
* We train our model on a single Nvidia Testla V100 machine. You can run ``bash run_train.sh`` to train the model with the default settings. 
* After the training procedure, you can run ``bash run_test.sh`` to test the model. 
* The training and testing data (DuConv or DuRecDial) can be changed through the argument ``data`` in both ``run_train.sh`` and ``run_test.sh``.  
