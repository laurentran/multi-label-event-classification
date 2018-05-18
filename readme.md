# Multi-label classification 

We train a ConvNet (CNN) to detect events in an 8x8 grid of measurements.  To train a multi-label classifier, we use multi-hot label encodings with a final sigmoid output layer.  

Update [train.py](https://github.com/laurentran/multi-label-event-classification/blob/master/train.py) to point to your data sources (both train and test sets), as well as multi-hot encoded labels.  Your labels should contain the same number of dimensions as the number of labels.  Edit the number of classes for your use case, and point to your output path for serializing the model.

For scoring, [score.py](https://github.com/laurentran/multi-label-event-classification/blob/master/score.py) deserializes the model saved during training and calls the model on a new input vector.  Point to your model file, and make edits as appropriate for your scoring workflow.
