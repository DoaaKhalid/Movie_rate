import numpy as np
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold


class Linear_Regression:
    model = linear_model.LinearRegression()

    def __init__(self, X_data, Y_data):
        Y_data = np.expand_dims(Y_data, axis=1)
        self.Xdata = X_data
        self.Ydata = Y_data

    def FitModel(self):
        self.model.fit(self.Xdata.T, self.Ydata)

    def TrainModel(self):
        crossvalidation = KFold(n_splits=5, random_state=None, shuffle=True)
        scores = cross_val_score(self.model, self.Xdata.T, self.Ydata, scoring="neg_mean_squared_error",
                                 cv=crossvalidation)
        print("Linear Regression Model "+"\nNumber of Parts Data: " + str(len(scores)) + "\nMSE: " + str(np.mean(np.abs(scores))) + "\nSTD: " + str(
            np.std(scores))+"\n__________________________________________________________________")
        #print("Scores: "+ str(scores))
