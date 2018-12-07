from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()

X_train, X_test, Y_train, Y_test = train_test_split(digits.data, digits.target, test_size=0.4)

clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(X=X_train, y=Y_train)
y_pred = clf.predict(X_test)

print("accuracy=%f" % ((sum(1 for i in range(len(Y_test)) if y_pred[i] == Y_test[i])) / len(Y_test)))
