# Importimi i librarive
import pandas as pd
import os
import pydotplus
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.metrics import confusion_matrix, classification_report
import graphviz

path_file = os.path.join("../../dataset", "parametrat_ajri_dataset_excel.xlsx")

# Load database
dataset = pd.read_excel(path_file)
X=dataset.iloc[:, 0:5].values #put the input fields
y = dataset.iloc[:, 5].values #put the output field

print(dataset.isnull().sum()) #print number of missing values

#Data pre-processing
#encode of strings to int
labelencoder_X=LabelEncoder()
X[:, 0]=labelencoder_X.fit_transform(X[:, 0])
labelencoder_X=LabelEncoder() 
X[:, 1]=labelencoder_X.fit_transform(X[:, 1])
labelencoder_X=LabelEncoder()
X[:, 2]=labelencoder_X.fit_transform(X[:, 2])
labelencoder_X=LabelEncoder()
X[:, 2]=labelencoder_X.fit_transform(X[:, 3])
labelencoder_X=LabelEncoder()
X[:, 2]=labelencoder_X.fit_transform(X[:, 4])

labelencoder_y=LabelEncoder() 
y=labelencoder_y.fit_transform(y)


# Split database in test and train group of data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 42)


# Fitting classifier to the Training set
model = DecisionTreeClassifier(criterion="entropy")
model = DecisionTreeRegressor()
model.fit(X_train,y_train)

# predicting values of Test group
y_pred = model.predict(X_test)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)


# convert to txt and dot file
with open("decision_tree.txt", "w") as f:
    f = export_graphviz(model, out_file=f)

def visualize_tree(tree, feature_names):
    """Create tree png using graphviz.

    Args
    ----
    tree -- scikit-learn DecsisionTree.
    feature_names -- list of feature names from X array
    """
    with open("dt.dot", 'w') as f:
        export_graphviz(tree, out_file=f,
                        feature_names=feature_names)

    command = ["dot", "-Tpng", "decision_tree.dot", "-o", "decision_tree.png"]

visualize_tree(model,["pm2_5","pm10","temp", "trysnia", "lageshtira"])

print('Confusion matrix: ')
print(cm)
print('Accuracy of decision tree classifier on test set: {:.2f}'.format(model.score(X_test, y_test)))

#Classification report, metrics
print(classification_report(y_test, y_pred))

#vizualise decision tree in png

feature_names = ["pm2_5","pm10","temp", "trysnia", "lageshtira"]
export_graphviz(model, out_file="decision_tree.dot", feature_names=feature_names)

with open("decision_tree.dot") as f:
    dot_graph = f.read()

graphviz.Source(dot_graph)
graph = pydotplus.graph_from_dot_data(dot_graph)
graph.write_png('decision_tree.png')

