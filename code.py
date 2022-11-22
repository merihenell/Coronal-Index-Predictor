import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

df_1 = pd.read_csv('sunspot_number.csv', sep = ';', names = ['Year', 'Month', 'Day', 'Frac', 'Sunspot number', 'Std', 'Obs', 'Ind'], header = None)

data_1 = df_1.assign(Date = df_1["Year"].astype(str) + '-' + df_1["Month"].astype(str) + '-' + df_1["Day"].astype(str))
data_1 = data_1.drop(['Year', 'Month', 'Day', 'Frac', 'Std', 'Obs', 'Ind'], axis=1)
data_1 = data_1[['Date', 'Sunspot number']]

df_2 = pd.read_csv('coronal_index.txt', delim_whitespace = True, names = ['Year', 'Month', 'Day', 'Coronal index'], header = None)

data_2 = df_2.assign(Date = df_2["Year"].astype(str) + '-' + df_2["Month"].astype(str) + '-' + df_2["Day"].astype(str))
data_2 = data_2.drop(['Year', 'Month', 'Day'], axis=1)
data_2 = data_2[['Date', 'Coronal index']]

data = pd.merge(data_1, data_2, on = 'Date')

plt.figure(figsize = (10, 6))

plt.scatter(data['Sunspot number'], data['Coronal index'], s = 10)
plt.xlabel('Sunspot number', size = 10)
plt.ylabel('Coronal index', size = 10)
plt.title('Coronal index vs Sunspot number', size = 12)

plt.show()

X = data['Sunspot number'].to_numpy().reshape(-1, 1)
y = data['Coronal index'].to_numpy()

# One feature

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size = 0.5, random_state = 42)

degrees = [1, 2, 3, 4, 5, 6]

tr_errors = []          
val_errors = []
test_errors = []

plt.figure(figsize = (9, 40))

for i, degree in enumerate(degrees):
    plt.subplot(len(degrees), 1, i + 1)
    
    lin_regr = LinearRegression(fit_intercept = False)
    poly = PolynomialFeatures(degree = degree)
    X_train_poly = poly.fit_transform(X_train)
    lin_regr.fit(X_train_poly, y_train)
    
    y_pred_train = lin_regr.predict(X_train_poly)
    tr_error = mean_squared_error(y_train, y_pred_train)
    X_val_poly = poly.transform(X_val)
    y_pred_val = lin_regr.predict(X_val_poly)
    val_error = mean_squared_error(y_val, y_pred_val)
    X_test_poly = poly.transform(X_test)
    y_pred_test = lin_regr.predict(X_test_poly)
    test_error = mean_squared_error(y_test, y_pred_test)
    
    tr_errors.append(tr_error)
    val_errors.append(val_error)
    test_errors.append(test_error)
    
    X_fit = np.linspace(0, 495)
    plt.tight_layout()
    plt.plot(X_fit, lin_regr.predict(poly.transform(X_fit.reshape(-1, 1))), label = 'Model')
    plt.scatter(X_train, y_train, color = 'b', s = 10, label = 'Training data')
    plt.scatter(X_val, y_val, color = 'r', s = 10, label = 'Validation data')
    plt.xlabel('Sunspot number', size = 10)
    plt.ylabel('Coronal index', size = 10)
    plt.legend(loc = 'best')
    plt.title(f'Polynomial degree = {degree}\nTraining error = {tr_error:.5}\nValidation error = {val_error:.5}', size = 12)

plt.show()

errors = {'Polynomial degree': degrees, 'Training error': tr_errors, 'Validation error': val_errors, 'Test error': test_errors}

pd.DataFrame({key: pd.Series(value) for key, value in errors.items()})

plt.figure(figsize = (10, 6))

plt.plot(degrees, tr_errors, color = 'b', label = 'Training error')
plt.plot(degrees, val_errors, color = 'r', label = 'Validation error')
plt.legend(loc = 'upper left')
plt.xlabel('Polynomial degree', size = 10)
plt.ylabel('Mean squared error', size = 10)
plt.title('Training and validation errors with one feature', size = 12)

plt.show()

# Five features

data['Pre 1'] = data['Sunspot number'].shift(1)
data['Pre 2'] = data['Sunspot number'].shift(2)
data['Pre 3'] = data['Sunspot number'].shift(3)
data['Pre 4'] = data['Sunspot number'].shift(4)

data = data.iloc[4:]

X = np.transpose(np.array([data['Sunspot number'], data['Pre 1'], data['Pre 2'], data['Pre 3'], data['Pre 4']]))
y = data['Coronal index'].to_numpy()

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size = 0.4, random_state = 42)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size = 0.5, random_state = 42)

degrees = [1, 2, 3, 4, 5, 6]    

tr_errors = []          
val_errors = []
test_errors = []

for i, degree in enumerate(degrees):
    lin_regr = LinearRegression(fit_intercept = False)
    poly = PolynomialFeatures(degree = degree)
    X_train_poly = poly.fit_transform(X_train)
    lin_regr.fit(X_train_poly, y_train)
    y_pred_train = lin_regr.predict(X_train_poly)
    tr_error = mean_squared_error(y_train, y_pred_train)
    X_val_poly = poly.transform(X_val)
    y_pred_val = lin_regr.predict(X_val_poly)
    val_error = mean_squared_error(y_val, y_pred_val)
    X_test_poly = poly.transform(X_test)
    y_pred_test = lin_regr.predict(X_test_poly)
    test_error = mean_squared_error(y_test, y_pred_test)
    
    tr_errors.append(tr_error)
    val_errors.append(val_error)
    test_errors.append(test_error)

errors = {'Polynomial degree': degrees, 'Training error': tr_errors, 'Validation error': val_errors, 'Test error': test_errors}

pd.DataFrame({key: pd.Series(value) for key, value in errors.items()})

plt.figure(figsize = (10, 6))

plt.plot(degrees, tr_errors, color = 'b', label = 'Training error')
plt.plot(degrees, val_errors, color = 'r', label = 'Validation error')
plt.legend(loc = 'upper left')
plt.xlabel('Polynomial degree', size = 10)
plt.ylabel('Mean squared error', size = 10)
plt.title('Training and validation errors with five features', size = 12)

plt.show()