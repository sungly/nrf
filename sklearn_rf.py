from extract import extract_data
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics


'''
This class uses sklearn library to determine the base accruacy of a generic
random forest classifier. 

Result accuracy = 81% 
'''

class SKRandomForest():
    def __init__(self):
        data, target = extract_data()
        
        # splits the data into training and testing 
        X_train, X_test, Y_train, Y_test = \
            train_test_split(data, target, test_size=0.3)
        
        clf = RandomForestClassifier(n_estimators=500)

        clf.fit(X_train, Y_train)

        y_pred = clf.predict(X_test)

        print("Accuracy: ", metrics.accuracy_score(Y_test, y_pred))


RandomForest()