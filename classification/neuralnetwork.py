# neuralnetwork.py
# -------------

# Neural network implementation
import util
import math
import numpy
PRINT = True

class NeuralNetworkClassifier:
  """
  neural network classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__( self, legalLabels, max_iterations):
    self.legalLabels = legalLabels
    self.type = "neuralnetwork"
    self.max_iterations = max_iterations
    self.weights = {}
    for label in legalLabels:
      self.weights[label] = util.Counter() # this is the data-structure you should use

  def setWeights(self, weights):
    assert len(weights) == len(self.legalLabels)
    self.weights == weights

  def sigmoidFunction(self, x):
    tempval = 1 + numpy.exp(-x)
    tempval = 1/tempval
    return tempval
      
  def train( self, trainingData, trainingLabels, validationData, validationLabels ):
    """
    The training loop for the neural network passes through the training data several
    times and updates the weight vector for each label based on classification errors.
    See the project description for details. 
    
    Use the provided self.weights[label] data structure so that 
    the classify method works correctly. Also, recall that a
    datum is a counter from features to values for those features
    (and thus represents a vector a values).
    """
    
    self.features = trainingData[0].keys() # could be useful later
    # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
    # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.
    
    for iteration in range(self.max_iterations):
      print "Starting iteration ", iteration, "..." 
      for i in range(len(trainingData)):
        "*** YOUR CODE HERE ***"
          
        scoreKeeper = util.Counter() # Keeps track of scores for each y'
        for crnt in self.legalLabels:
          scoreKeeper[crnt] = None
          
        for label in self.legalLabels:
          temp = trainingData[i] * self.weights[label]
          temp = self.sigmoidFunction(temp)
          # if the calculated score is greater then score stored for that y', it is replaced
          if temp > scoreKeeper[label] or scoreKeeper[label] is None: 
            scoreKeeper[label] = temp 

        if scoreKeeper.argMax() != trainingLabels[i]:
          self.weights[trainingLabels[i]] += trainingData[i]
          self.weights[scoreKeeper.argMax()] -= trainingData[i]

        # Get the y' with max score, and check if that y' matches with y

          
    
  def classify(self, data ):
    """
    Classifies each datum as the label that most closely matches the prototype vector
    for that label.  See the project description for details.
    
    Recall that a datum is a util.counter... 
    """
    guesses = []
    for datum in data:
      vectors = util.Counter()
      for l in self.legalLabels:
        vectors[l] = self.weights[l] * datum
      guesses.append(vectors.argMax())
    return guesses

  
  def findHighWeightFeatures(self, label):
    """
    Returns a list of the 100 features with the greatest weight for some label
    """
    featuresWeights = []

    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresWeights

