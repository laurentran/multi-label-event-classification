from keras.models import load_model
import numpy as np
import sys

# this file is illustrative of how to deserialize the saved model and make a prediction
# adjustments should be made based on how data gets passed in and how it should be written to the DB
def main(X):
    model = load_model('outputs/model.h5') # edit model path as appropriate
    prediction = model.predict(X) # prediction will be a vector of probabilities for each event
    print(prediction) # instead of printing, need to write to Cosmos DB

if __name__ == "__main__":
    # the argument passed in should be the new feature vector of 64 values
    # it may not be in the correct shape for the model to make a prediction
    # so, in main(), the vector should be reshaped as a (1,8,8,1) numpy array for the model to make an inference
    main(sys.argv[1]) 