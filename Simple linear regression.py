import numpy as np

from sklearn.linear_model import LinearRegression

x = np.array([20,43,63,26,53,31,58, 46 ,58,70,46,53,70,20,63,43,26,19,31,23]).reshape((-1, 1))

y = np.array([120,128,141,126,134,128,136,132,140,144,128,136,146,124,143,130,124,121,126,123])

model = LinearRegression().fit(x, y)

r_sq = model.score(x, y)

print('coefficient of determination:', r_sq)

print('intercept:', model.intercept_)

print('slope:', model.coef_)

new_model = LinearRegression().fit(x, y.reshape((-1, 1)))

print('intercept:', new_model.intercept_)

print('slope:', new_model.coef_)

y_pred = model.predict(x)

print('predicted response:', y_pred, sep='\n')

y_pred = model.intercept_ + model.coef_ * x

print('predicted response:', y_pred, sep='\n')

x_new = np.arange(5).reshape((-1, 1))

print(x_new)

y_new = model.predict(x_new)

print(y_new)

import matplotlib.pyplot as plt 

plt.plot( x,  model.intercept_ +new_model.coef_* x  )

plt.scatter(x, y, c ="red") 

plt.title(" regression curve for Age vs B.P.",fontsize=26)

plt.xlabel("  age", fontsize= 20)

plt.ylabel(" Blood presuure in mmHg",fontsize = 20)

plt.show()
