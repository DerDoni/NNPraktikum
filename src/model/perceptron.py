# -*- coding: utf-8 -*-

import sys
import logging

import numpy as np

from util.activation_functions import Activation
from model.classifier import Classifier

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.DEBUG,
                    stream=sys.stdout)


class Perceptron(Classifier):
    """
    A digit-7 recognizer based on perceptron algorithm

    Parameters
    ----------
    train : list
    valid : list
    test : list
    learningRate : float
    epochs : positive int

    Attributes
    ----------
    learningRate : float
    epochs : int
    trainingSet : list
    validationSet : list
    testSet : list
    weight : list
    """
    def __init__(self, train, valid, test, 
                                    learningRate=0.01, epochs=50):

        self.learningRate = learningRate
        self.epochs = epochs

        self.trainingSet = train
        self.validationSet = valid
        self.testSet = test

        # Initialize the weight vector with small random values
        # around 0 and0.1
        self.weight = np.random.rand(self.trainingSet.input.shape[1])/100

    def train(self, verbose=True):
        """Train the perceptron with the perceptron learning algorithm.

        Parameters
        ----------
        verbose : boolean
            Print logging messages with validation accuracy if verbose is True.
        """
        minError = 1000
        # As long as max Step not reached
        for i in range(0, self.epochs):
            #Compute which elements are wrongly classified
            error = self.classify(self.trainingSet.input)-self.trainingSet.label
            self.updateWeights(self.trainingSet.input, error)
            #Compute the error in the validation set
            errorV = np.sum(self.classify(self.validationSet.input)!=self.validationSet.label)
            if(errorV<=minError):
                minWeights = self.weight
                minError = errorV
            if(verbose):
                print "Absolute error in epoch {}: {}".format(i, errorV)
        self.weight = minWeights
		
    def classify(self, testInstance):
        """Classify a single instance.

        Parameters
        ----------
        testInstance : list of floats

        Returns
        -------
        bool :
            True if the testInstance is recognized as a 7, False otherwise.
        """
        return self.fire(testInstance)

    def evaluate(self, test=None):
        """Evaluate a whole dataset.

        Parameters
        ----------
        test : the dataset to be classified
        if no test data, the test set associated to the classifier will be used

        Returns
        -------
        List:
            List of classified decisions for the dataset's entries.
        """
        if test is None:
            test = self.testSet.input
        # Once you can classify an instance, just use map for all of the test
        # set.
        return list(map(self.classify, test))

    def updateWeights(self, input, error):
        # Write your code to update the weights of the perceptron here
        sumErr = np.matmul(np.transpose(input),error)
        self.weight = self.weight - self.learningRate*sumErr
         
    def fire(self, input):
        """Fire the output of the perceptron corresponding to the input """
        return Activation.sign(np.matmul(np.array(input),np.array(self.weight)))
