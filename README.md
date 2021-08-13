# Assignment: 

* TASK 1: Train BERT using the code mentioned [here](https://drive.google.com/file/d/1Zp2_Uka8oGDYsSe5ELk-xz6wIX8OIkB7/view)  on the Squad Dataset for 20% overall samples (1/5 Epochs). Show results on 5 samples. 
* TASK 2: Reproduce [these](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) results, and show output on 5 samples.
* TASK 3: Reproduce the training explained in this [blog](https://towardsdatascience.com/bart-for-paraphrasing-with-simple-transformers-7c9ea3dfdd8c). You can decide to pick fewer datasets. 


## Task 1

### BERT
Bidirectional Encoder Representation from Transformer (BERT), as the name suggests, is based on the transformer model. We can perceive BERT as the transformer, but only with the encoder.

We  feed the sentence as input to the transformer's encoder and it returns the representation for each word in the sentence as an output. Well, that's exactly what BERT is – an Encoder Representation from Transformer. Okay, so what about the term Bidirectional?

The encoder of the transformer is bidirectional in nature since it can read a sentence in both directions. Thus, BERT is basically the Bidirectional Encoder Representation obtained from the Transformer.

Let's understand how BERT is bidirectional encoder representation from the transformer with the help of an example. Let's take the same sentences we saw in the previous section.

Say we have a sentence A: 'He got bit by Python'. Now, we feed this sentence as an input to the transformer's encoder and get the contextual representation (embedding) of each word in the sentence as an output. Once we feed the sentence as an input to the encoder, the encoder understands the context of each word in the sentence using the multi-head attention mechanism (relates each word in the sentence to all the words in the sentence to learn the relationship and contextual meaning of words) and returns the contextual representation of each word in the sentence as an output.

![image](https://user-images.githubusercontent.com/46154140/129351512-26530dbc-7b1d-4365-9dd2-6d93616ccb18.png)


### Training Log

![image](https://user-images.githubusercontent.com/46154140/129351848-7ce34b2f-028b-4fb0-aa53-a75c454fe70f.png)

    Training Loss 1.251854658126831


### Evaluation Results
```
Evaluating:   0%|          | 0/425 [00:00<?, ?it/s]***** Running evaluation *****
  Num examples = 13600
  Batch size = 32
Evaluating: 100%|██████████| 425/425 [03:37<00:00,  1.95it/s]
{
  "exact": 49.38937084140487,
  "f1": 54.0109620487612,
  "total": 11873,
  "HasAns_exact": 64.6255060728745,
  "HasAns_f1": 73.88194203862014,
  "HasAns_total": 5928,
  "NoAns_exact": 34.19680403700589,
  "NoAns_f1": 34.19680403700589,
  "NoAns_total": 5945,
  "best_exact": 57.01170723490272,
  "best_exact_thresh": -5.869788765907288,
  "best_f1": 59.45140875382585,
  "best_f1_thresh": -5.801982164382935
}
```

### Model Prediction on random test set

******************************************************************************************

question         >> What was the Grand 1401 building renamed as?

Answer BY Model  >> San Joaquin Light & Power Building

******************************************************************************************
******************************************************************************************

question         >> What is the result of rebellion according to Black's Law Dictionary?

Answer BY Model  >> non-violence

******************************************************************************************
******************************************************************************************

question         >> Who was the first geologist?

Answer BY Model  >> James Hutton

******************************************************************************************
******************************************************************************************

question         >> What does not regularly use input coding as its concrete choice?

Answer BY Model  >> 

******************************************************************************************
**********************************************************************************************************************************

question         >> Evolution of what part of the immune system occurred in the evolutionary ancestor of jawed vertebrates?

Answer BY Model  >> adaptive immune system

******************************************************************************************
******************************************************************************************

question         >> What is the virus in humans that causes cervical cancer?

Answer BY Model  >> papillomavirus

******************************************************************************************
******************************************************************************************

question         >> What is included in Medication Therapy Management?

Answer BY Model  >> clinical services

******************************************************************************************
******************************************************************************************

question         >> How did Yale introduce a new era in football?

Answer BY Model  >> evolution of the college game

******************************************************************************************
******************************************************************************************

question         >> What river does Berlin straddle?

Answer BY Model  >> Vistula River

******************************************************************************************
******************************************************************************************

question         >> How many nations are within the Amazon Basin?

Answer BY Model  >> 

******************************************************************************************



## Task 2

## BERT Fine-Tuning With Pytorch

### Training Logs -
```

======== Epoch 1 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:26.
  Batch    80  of    241.    Elapsed: 0:00:52.
  Batch   120  of    241.    Elapsed: 0:01:18.
  Batch   160  of    241.    Elapsed: 0:01:45.
  Batch   200  of    241.    Elapsed: 0:02:11.
  Batch   240  of    241.    Elapsed: 0:02:37.

  Average training loss: 0.50
  Training epcoh took: 0:02:37

Running Validation...
  Accuracy: 0.78
  Validation Loss: 0.49
  Validation took: 0:00:06

======== Epoch 2 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:26.
  Batch    80  of    241.    Elapsed: 0:00:52.
  Batch   120  of    241.    Elapsed: 0:01:18.
  Batch   160  of    241.    Elapsed: 0:01:45.
  Batch   200  of    241.    Elapsed: 0:02:11.
  Batch   240  of    241.    Elapsed: 0:02:37.

  Average training loss: 0.31
  Training epcoh took: 0:02:37

Running Validation...
  Accuracy: 0.79
  Validation Loss: 0.53
  Validation took: 0:00:06

======== Epoch 3 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:26.
  Batch    80  of    241.    Elapsed: 0:00:52.
  Batch   120  of    241.    Elapsed: 0:01:18.
  Batch   160  of    241.    Elapsed: 0:01:45.
  Batch   200  of    241.    Elapsed: 0:02:11.
  Batch   240  of    241.    Elapsed: 0:02:37.

  Average training loss: 0.20
  Training epcoh took: 0:02:37

Running Validation...
  Accuracy: 0.80
  Validation Loss: 0.61
  Validation took: 0:00:06

======== Epoch 4 / 4 ========
Training...
  Batch    40  of    241.    Elapsed: 0:00:26.
  Batch    80  of    241.    Elapsed: 0:00:52.
  Batch   120  of    241.    Elapsed: 0:01:18.
  Batch   160  of    241.    Elapsed: 0:01:44.
  Batch   200  of    241.    Elapsed: 0:02:10.
  Batch   240  of    241.    Elapsed: 0:02:36.

  Average training loss: 0.14
  Training epcoh took: 0:02:37

Running Validation...
  Accuracy: 0.80
  Validation Loss: 0.66
  Validation took: 0:00:06

Training complete!
Total training took 0:10:51 (h:mm:ss)
```
### Training Stats - 

|Epoch|Training Loss|	Valid. Loss|	Valid. Accur.|	Training Time|	Validation Time|
|-----|-------------|------------|-------------|-----------------|-----------------|		
|1	|0.50|	0.49|	0.78|	0:02:37|	0:00:06|
|2|	0.31|	0.53|	0.79|	0:02:37|	0:00:06|
|3|	0.20|	0.61|	0.80|	0:02:37|	0:00:06|
|4|	0.14	|0.66	|0.80	|0:02:37	|0:00:06|

### Training and Validation Loss

![image](https://user-images.githubusercontent.com/46154140/129196165-87b587a8-fef4-4adc-acac-f17eb4b2862a.png)


### Mathews Correlation Coeffecient Per Batch

![image](https://user-images.githubusercontent.com/46154140/129196367-05655307-b77d-4bf7-a92f-fc30128020a0.png)

**Total MCC: 0.560**

### Classification Report
```
              precision    recall  f1-score   support

           0       0.81      0.56      0.66       162
           1       0.82      0.94      0.88       354

    accuracy                           0.82       516
   macro avg       0.82      0.75      0.77       516
weighted avg       0.82      0.82      0.81       516

```

### Confusion Matrix

![image](https://user-images.githubusercontent.com/46154140/129196869-78b9f940-8209-4ff6-885c-2bd51118d613.png)


### Prediction On 5 Random Samples

******************************************************************************************
Sentence ---  I like Bill's yellow shirt, but not Max's.

Actual Label >>>>    1

Predicted Label >>>    1


******************************************************************************************
******************************************************************************************
Sentence ---  Chocolate eggs were hidden from each other by the children.

Actual Label >>>>    0

Predicted Label >>>    1


******************************************************************************************
******************************************************************************************
Sentence ---  The paper was written by John up.

Actual Label >>>>    0

Predicted Label >>>    1


******************************************************************************************
******************************************************************************************
Sentence ---  The bucket was kicked by Pat.

Actual Label >>>>    1

Predicted Label >>>    1


******************************************************************************************
******************************************************************************************
Sentence ---  I never know which papers Sandy has read, but I usually know how many.

Actual Label >>>>    1

Predicted Label >>>    1


******************************************************************************************



## Task - 3

### BART






