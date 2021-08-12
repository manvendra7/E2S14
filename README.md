## TASK 2: 
Reproduce [these](https://mccormickml.com/2019/07/22/BERT-fine-tuning/) results, and show output on 5 samples.

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








