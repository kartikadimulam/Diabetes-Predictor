import numpy as np
import matplotlib.pyplot as plt

styles = plt.style.available
#print(styles)

plt.style.use('seaborn-v0_8-whitegrid')

from sklearn import linear_model, datasets

diabetes = datasets.load_diabetes()
print(diabetes.DESCR)

diabetes_X = diabetes.data[:, np.newaxis, 2]
print(diabetes_X)

diabetes_X_training = diabetes_X[:-20]
diabetes_X_testing = diabetes_X[-20:]

diabetes_y = diabetes.target

diabetes_y_training = diabetes_y[:-20]
diabetes_y_testing = diabetes_y[-20:]

regression = linear_model.LinearRegression()
#object instance
regression.fit(diabetes_X_training, diabetes_y_training)

mse = (regression.predict(diabetes_X_testing)-diabetes_y_testing)**2

print('Mean Squared Error is %2f' % np.mean(mse))

print('Variance Score: %2f' % regression.score(diabetes_X_testing, diabetes_y_testing))

plt.figure(figsize=(10,7))
plt.scatter(diabetes_X_testing, diabetes_y_testing, color='black')
plt.plot(diabetes_X_testing, regression.predict(diabetes_X_testing), color='blue')

plt.title('Linear Regression Model', fontsize=14)
plt.xlabel('Independent Variable', fontsize=12)
plt.ylabel('Dependent Variable', fontsize=12)
plt.show()


