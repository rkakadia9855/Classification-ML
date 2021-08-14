# naiveBayes.py
# -------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
import classificationMethod
import math

class NaiveBayesClassifier(classificationMethod.ClassificationMethod):
  """
  See the project description for the specifications of the Naive Bayes classifier.
  
  Note that the variable 'datum' in this code refers to a counter of features
  (not to a raw samples.Datum).
  """
  def __init__(self, legalLabels):
    self.legalLabels = legalLabels
    self.type = "naivebayes"
    self.k = 1 # this is the smoothing parameter, ** use it in your train method **
    self.automaticTuning = False # Look at this flag to decide whether to choose k automatically ** use this in your train method **
    
  def setSmoothing(self, k):
    """
    This is used by the main method to change the smoothing parameter before training.
    Do not modify this method.
    """
    self.k = k

  def train(self, trainingData, trainingLabels, validationData, validationLabels):
    """
    Outside shell to call your method. Do not modify this method.
    """  
      
    # might be useful in your code later...
    # this is a list of all features in the training set.
    self.features = list(set([ f for datum in trainingData for f in datum.keys() ]));
    
    if (self.automaticTuning):
        kgrid = [0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 20, 50]
    else:
        kgrid = [self.k]
        
    self.trainAndTune(trainingData, trainingLabels, validationData, validationLabels, kgrid)
      
  def trainAndTune(self, trainingData, trainingLabels, validationData, validationLabels, kgrid):
    """
    Trains the classifier by collecting counts over the training data, and
    stores the Laplace smoothed estimates so that they can be used to classify.
    Evaluate each value of k in kgrid to choose the smoothing parameter 
    that gives the best accuracy on the held-out validationData.
    
    trainingData and validationData are lists of feature Counters.  The corresponding
    label lists contain the correct label for each datum.
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """

    "*** YOUR CODE HERE ***"

    priorDistributor = util.Counter() # prior distribution of labels
    featureConditionalProb = util.Counter() # Conidtional probability of a feature = 1 for a label
    featureCounterY = util.Counter() # counts individual feature for a label y

    for i in range(len(trainingData)):
      datum = trainingData[i]
      label = trainingLabels[i]
      priorDistributor[label] += 1 # increase the counter of a label
      for feature, val in datum.items():
        featureCounterY[(feature,label)] += 1 # If a particular feature appears, increase its counter regardless of feature value
        if val > 0: 
          featureConditionalProb[(feature, label)] += 1 # Now increase its count in conditional probability if its val = 1 or 2

    # Prepare prior dictionary to calculate the log(P(y))
    self.prior = util.Counter()
    for key, val in priorDistributor.items():
      self.prior[key] += val
    # Convert prior distribution in terms of log
    for key in self.prior:
      self.prior[key] = math.log(self.prior[key])

    mostAccurateK = -1 # keeps track of which k value produced most accurate results
    # The loop below estimates P(Fi = fi | Y = y) and applies smoothing to it (According to berkeley documentation)
    for k in kgrid: 
      featureEqualsOne = util.Counter() # c(fi, y)+k
      totalFeatureCount = util.Counter() # Summation(c(!fi, y)+k)

      # Get the values for proceeding to the step where we estimate P(fi | y)
      for key, val in featureCounterY.items():
        totalFeatureCount[key] += val
      for key, val in featureConditionalProb.items():
        featureEqualsOne[key] += val

      # Add k to smooth the values
      for label in self.legalLabels:
        for feature in self.features:
          featureEqualsOne[ (feature, label)] +=  k
          totalFeatureCount[(feature, label)] +=  2*k 

      # Time to finally calculate P(fi | y)
      for x, count in featureEqualsOne.items():
        featureEqualsOne[x] = float(count) / float(totalFeatureCount[x])

      self.finalConditionalProbability = featureEqualsOne

      # This is for autotune. It keeps track of which k value produced best accuracy
      predictions = self.classify(validationData)
      accuracyScore =  [predictions[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
      bestParams = (featureEqualsOne, k) if accuracyScore > mostAccurateK else bestParams
      mostAccurateK = accuracyScore if accuracyScore > mostAccurateK else mostAccurateK

    self.finalConditionalProbability, self.k = bestParams
    
    
        
  def classify(self, testData):
    """
    Classify the data based on the posterior distribution over labels.
    
    You shouldn't modify this method.
    """
    guesses = []
    self.posteriors = [] # Log posteriors are stored for later data analysis (autograder).
    for datum in testData:
      posterior = self.calculateLogJointProbabilities(datum)
      guesses.append(posterior.argMax())
      self.posteriors.append(posterior)
    return guesses
      
  def calculateLogJointProbabilities(self, datum):
    """
    Returns the log-joint distribution over legal labels and the datum.
    Each log-probability should be stored in the log-joint counter, e.g.    
    logJoint[3] = <Estimate of log( P(Label = 3, datum) )>
    
    To get the list of all possible features or labels, use self.features and 
    self.legalLabels.
    """
    logJoint = util.Counter()
    
    "*** YOUR CODE HERE ***"

    for label in self.legalLabels:
      logJoint[label] = self.prior[label]
      for feature, value in datum.items():
        # Calculating {log(P(y) + Sum(log(P(fi | y))))}
        if value != 0:
          logJoint[label] += math.log(self.finalConditionalProbability[feature,label])
        else:
          logJoint[label] += math.log(1-self.finalConditionalProbability[feature,label])
          
    return logJoint
  
  def findHighOddsFeatures(self, label1, label2):
    """
    Returns the 100 best features for the odds ratio:
            P(feature=1 | label1)/P(feature=1 | label2) 
    
    Note: you may find 'self.features' a useful way to loop through all possible features
    """
    featuresOdds = []
       
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

    return featuresOdds