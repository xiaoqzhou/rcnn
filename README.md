# rcnn
This code is the program (RCNN) of the paper "Answer Sequence Learning with Neural Networks for Answer Selection in Community Question Answering" ACL-IJCNLP 2015: http://www.aclweb.org/anthology/P15-2117

%python  Theano

The training and test for RCNN model:
	python model_train.py
	python model_test.py

The corresponding discription of each .py:

  model_train.py        - Model training   (the model saved in ./WeightModel)
  model_test.py         - Model test       (the result saved in ./Result)

  BidirectLstm.py       - The Class of Bi-RNNs
  LeNetConvPoolLayer.py - The Class of CNN
  mlp.py                - The Class of Hidden layer (fully connect layer)
  ReLU.py               - The activation function ReLu (Rectified Linear Units)

  parameters.py         - Intilization of model parameters
  prepare.py            - The Class of the preprocessing of QA data

  Model Evaluation:
  1 get-label.py          - convert the result of RCNN model to the answer label for evaluation (saved in ./Label/)  
  2 CQA-QL-test-gold      - the gold label of test data
  3 scoreA.pl             - the tool of computing the performance (provided by the answer selection task of semevel2015)

The Initialization of wordembedding is trained with the raw data in http://alt.qcri.org/semeval2015/task3/.
