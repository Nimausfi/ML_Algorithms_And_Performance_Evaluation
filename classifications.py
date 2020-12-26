from sklearn import tree
from sklearn import svm
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier



def DecisionTree(x, y, split):
    x, y = oneHotEncode(x, y)
 

    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = tree.DecisionTreeClassifier()
    print(len(training_x))
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1score = (2*precision*recall)/(precision+recall)
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))


def oneHotEncode(x, y):
    distinct_words = []
    for i in range(0, len(x)):
        for j in range(0, len(x[i])):
            if x[i][j] not in distinct_words:
                distinct_words.append(x[i][j])
    print(len(distinct_words))
    all_encode = []
    new_y = []
    for i in range(0, len(x)):
        encode = []
        for j in distinct_words:
            encode.append(0)
        for j in range(0, len(x[i])):
            for k in range(0, len(distinct_words)):
                if x[i][j] == distinct_words[k]:
                    encode[k] = 1
                    break
        all_encode.append(encode)
        if y[i] == "Y":
            new_y.append(1)
        else:
            new_y.append(0)
    return all_encode, new_y
 



def SVM(x, y, split):
    x, y = oneHotEncode(x, y)
    

    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = svm.SVC()
    print(len(training_x))
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1score = (2*precision*recall)/(precision+recall)
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))




def SGD(x, y, split):
    x, y = oneHotEncode(x, y)


    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = linear_model.SGDClassifier()

    print(len(training_x))
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    f1score = (2*precision*recall)/(precision+recall)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))




def RandomForest(x, y, split):
    x, y = oneHotEncode(x, y)
    

    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = RandomForestClassifier(random_state=0)

    print(len(training_x))
    # for i in training_x:
    #     print(i)
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    # print(predicted_y)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1score = (2*precision*recall)/(precision+recall)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))




def NaiveBayes(x, y, split):
    x, y = oneHotEncode(x, y)
    #new_y = []


    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = GaussianNB()

    print(len(training_x))
    # for i in training_x:
    #     print(i)
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    # print(predicted_y)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1score = (2*precision*recall)/(precision+recall)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))
    
    
    
    
def GradBoosting(x, y, split):
    x, y = oneHotEncode(x, y)
    

    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = GradientBoostingClassifier()

    print(len(training_x))
    # for i in training_x:
    #     print(i)
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    # print(predicted_y)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1score = (2*precision*recall)/(precision+recall)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))
    
    
    
    
def LogisticReg(x,y,split):
    x, y = oneHotEncode(x, y)


    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = linear_model.LogisticRegression()
    print(len(training_x))
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1score = (2*precision*recall)/(precision+recall)
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))
    

    
    
def KNeighbors(x, y, split):
    x, y = oneHotEncode(x, y)
    

    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = KNeighborsClassifier()

    print(len(training_x))
    # for i in training_x:
    #     print(i)
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    # print(predicted_y)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    accuracy = (TP+TN)/(TP+TN+FP+FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1score = (2*precision*recall)/(precision+recall)
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))        
    
    

    
def AdaBoost(x, y, split):
    x, y = oneHotEncode(x, y)
    
    
    training_length = int(len(x) * split)
    training_x = x[0: training_length]
    training_y = y[0: training_length]
    testing_x = x[training_length:]
    testing_y = y[training_length:]
    clf = AdaBoostClassifier()
    print(len(training_x))
    print(len(training_y))
    clf = clf.fit(training_x, training_y)
    predicted_y = clf.predict(testing_x)
    TP, TN, FP, FN = 0, 0, 0, 0
    for i in range(0, len(testing_y)):
        if testing_y[i] == 1 and predicted_y[i] == 1:
            TP += 1
        elif testing_y[i] == 1 and predicted_y[i] == 0:
            FP += 1
        elif testing_y[i] == 0 and predicted_y[i] == 0:
            TN += 1
        elif testing_y[i] == 0 and predicted_y[i] == 1:
            FN += 1
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)
    f1score = (2*precision*recall)/(precision+recall)
    print("TP: " + str(TP) + " FP: " + str(FP) + " TN: " + str(TN) + " FN: " + str(FN))
    print("Accuracy: " + str(accuracy))
    print("Precision: " + str(precision))
    print("Recall: " + str(recall))
    print("F-1 Score: " + str(f1score))    
    
    
