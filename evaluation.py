def evaluate(predictions, answers):
    index = range(len(predictions))
    correct = 0
    for i in index:
        if predictions[i] == answers[i]:
           correct += 1
    total = len(predictions)
    return (float(correct)/float(total)*100.)
