def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0
    
    TP = 0
    FN = 0
    FP = 0
    TN = 0
    
    for i in range(prediction.size):
        if (prediction[i] == True) and (ground_truth[i] == True):
            TP += 1
        elif (prediction[i] == False) and (ground_truth[i] == False):
            TN += 1
        elif (prediction[i] == False) and (ground_truth[i] == True):
            FP += 1
        elif (prediction[i] == True) and (ground_truth[i] == False):
            FN += 1
    accuracy = (TP + TN)/(TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    
    f1 = (2*TP)/(2*TP+FP+FN)
    # TODO: implement metrics!
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, f1, accuracy


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    # TODO: Implement computing accuracy
    
    correct = 0
    all_predictions = prediction.size
    
    for i in range(prediction.size):
        if(prediction[i] == ground_truth[i]):
            correct += 1
    
    accuracy = (correct / all_predictions)
    
    
    return accuracy
