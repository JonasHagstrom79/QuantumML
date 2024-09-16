from sklearn.metrics import recall_score, precision_score, confusion_matrix
import random
from qiskit import QuantumCircuit, transpile

def classifier_report(name, run, classify, input, labels,confusion_matrix, precision_score, recall_score):    
    """An reusable function to unmask the hypocrite classifier"""
    
    cr_predictions = run(classify, input)
    cr_cm = confusion_matrix(labels, cr_predictions)

    cr_precision = precision_score(labels, cr_predictions)
    cr_recall = recall_score(labels, cr_predictions)
    cr_specificity = specificity(cr_cm)
    cr_npv = npv(cr_cm)
    cr_level = 0.25*(cr_precision + cr_recall + cr_specificity + cr_npv)

    print('The precision score of the {} classifier is {:.2f}'.format(name, cr_precision))
    print('The recall score of the {} classifier is {:.2f}'.format(name, cr_recall))
    print('The specificity score of the {} classifier is {:.2f}'.format(name, cr_specificity))
    print('The npv score of the {} classifier is {:.2f}'.format(name, cr_npv))
    print('The information level is: {:.2f}'.format(cr_level))

# A generalized hypocrit classifier
def hypocrite(passanger, weight):
    """The hypocrite classifier takes the passenger data and a weight value. The weight
        is a number between −1 and 1. It denotes the classifier’s tendency to predict
        death (negative values) or survival (positive values).
        The formula weight*0.5+random.uniform(0, 1) generates numbers between −0.5
        and 1.5. The min and max functions ensure the result to be between 0 and 1. The
        round function returns either 0 (death) or 1 (survival).
        Depending on the weight, the chances to return one or the other prediction
        differs.
    """
    return round(min(1, max(0,weight*0.5+random.uniform(0,1))))

def specificity(matrix):
    """The function specificity takes the confusion matrix as a parameter (line 1). It
        divides the true negatives (matrix[0][0]) by the sum of the true negatives and'
        the false positives (matrix[0][1]) (line 2)."""
    return matrix[0][0]/(matrix[0][0]+matrix[0][1]) if (matrix[0][0]+matrix[0][1] > 0) else 0

def npv(matrix):
    """The function npv takes the confusion matrix as a parameter (line 4) and di-
        vides the true negatives by the sum of the true negatives and the false nega-
        tives (matrix[1][0])."""
    return matrix[0][0]/(matrix[0][0]+matrix[1][0]) if (matrix[0][0]+matrix[1][0] > 0) else 0

def evaluate(predictions, actual):
    correct = list(filter(
        lambda item: item[0] == item[1],
        list(zip(predictions,actual))
    ))
    return '{} correct predictions out of {}. Accuracy {:.0f} %'.format(len(correct), len(actual), 100*len(correct)/len(actual))

def load_data_from_csv():
    """Load the data from the csv-files"""
    import pandas as pd
    train = pd.read_csv('../data/train.csv')
    test = pd.read_csv('../data/test.csv')

def pqc_classify(backend, passenger_state):  
    """
    backend - a qiskit backend to run the quantum circuit at
    passanger_state - a valid quantum state vector
    """
    # Create a quantum circuit with one qbit and one classical bit
    qc = QuantumCircuit(1,1)

    # Define state |Psi> and initialize the circuit
    qc.initialize(passenger_state, 0)

    # Measure the qbit
    qc.measure(0,0)

    # Transpile the circuit for the backend
    qc_transpiled = transpile(qc, backend)

    # Run the quantum circuit
    job = backend.run(qc_transpiled, shots=1)
    result = job.result()

    # Get the counts, these are either {'0':1} or {'1':1}
    counts = result.get_counts(qc)

    # Get the bit 0 or 1
    return int(list(counts.keys())[0])

def run(f_classify, x):
    """Runs classifier"""
    return list(map(f_classify, x))

def classify(passenger):
    """Baseline model, random classifier"""    
    random.seed(a=None, version=2)
    return random.randint(0, 1)

def weigh_feature(feature, weight):
    """
    feature -- the single value of a passenger's feature
    weight -- the overall weight of this feature
    returns the weighted feature
    """
    return feature*weight

