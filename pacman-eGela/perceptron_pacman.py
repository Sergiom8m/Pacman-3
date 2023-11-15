# perceptron_pacman.py
# --------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# Perceptron implementation for apprenticeship learning
import util
from perceptron import PerceptronClassifier
from pacman import GameState

PRINT = True


class PerceptronClassifierPacman(PerceptronClassifier):
    def __init__(self, legalLabels, maxIterations):
        PerceptronClassifier.__init__(self, legalLabels, maxIterations)
        self.weights = util.Counter()

    def classify(self, data):
        """
        Data contains a list of (datum, legal moves)
        
        Datum is a Counter representing the features of each GameState.
        legalMoves is a list of legal moves for that GameState.
        """
        guesses = []
        for datum, legalMoves in data:
            vectors = util.Counter()
            for move in legalMoves:
                vectors[move] = self.weights * datum[move]  # changed from datum to datum[l]
            guesses.append(vectors.argMax())
        return guesses

    def train(self, trainingData, trainingLabels, validationData, validationLabels):
        self.features = trainingData[0][0]['Stop'].keys()  # could be useful later
        # DO NOT ZERO OUT YOUR WEIGHTS BEFORE STARTING TRAINING, OR
        # THE AUTOGRADER WILL LIKELY DEDUCT POINTS.

        for iteration in range(self.max_iterations):
            print("Starting iteration ", iteration, "...")
            for i in range(len(trainingData)):
                
                # Calcular la clase predicha
                best_label = self.classify([trainingData[i]])[0]

                # Obtener el valor de clase real
                real_label = trainingLabels[i]

                # Si la clase predicha no es correcta --> Actualizar pesos
                if best_label != real_label:

                    # Acercar a la clase real
                    self.weights = self.weights + trainingData[i][0][real_label]

                    # Alejar de la clase mal predicha
                    self.weights = self.weights - trainingData[i][0][best_label]

