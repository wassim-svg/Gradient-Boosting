from sklearn import datasets

diabetes = datasets.load_diabetes()

X = diabetes.data
y = diabetes.target

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=17, test_size=0.2)
from sklearn.ensemble import GradientBoostingRegressor

gbr = GradientBoostingRegressor()
gbr.fit(X_train,y_train)

y_pred = gbr.predict(X_test)

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
r2_score(y_test, y_pred)

param_grid={
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4,5],
    'min_samples_split': [2,3,4],
    'min_samples_leaf': [1,2,3]
}

from sklearn.model_selection import GridSearchCV

gs = GridSearchCV(gbr, param_grid, cv=3, n_jobs=-1, scoring='neg_mean_squared_error')
gs.fit(X_train, y_train)

y_pred = gs.predict(X_test)

mean_squared_error(y_test, y_pred)
mean_absolute_error(y_test, y_pred)
r2_score(y_test, y_pred)

gs.best_params_