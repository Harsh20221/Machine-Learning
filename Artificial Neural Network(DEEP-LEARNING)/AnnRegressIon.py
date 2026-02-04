import numpy as np
import tensorflow as tf
import pandas as pd

tf.__version__

#* ACCESSING THE DATASET
dataset = pd.read_csv('Churn_Modelling.csv')

#* SELECT INPUT FEATURES AND TARGET
x = dataset.iloc[:, 3:-1].values   # CreditScore to EstimatedSalary
y = dataset.iloc[:, -1].values    # Target (we will treat it as numerical output)

#* LABEL ENCODING (Gender Column)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:, 2] = le.fit_transform(x[:, 2])

#* ONE HOT ENCODING (Geography Column)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(
    transformers=[('encoder', OneHotEncoder(), [1])],
    remainder='passthrough'
)
x = np.array(ct.fit_transform(x))

#* SPLITTING INTO TRAIN AND TEST SET
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=0
)

#* FEATURE SCALING
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#* INITIALIZING THE ANN MODEL
ann = tf.keras.models.Sequential()

#* ADDING INPUT + HIDDEN LAYERS
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#* OUTPUT LAYER FOR REGRESSION
ann.add(tf.keras.layers.Dense(units=1, activation='linear'))

#* COMPILING THE MODEL
ann.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

#* TRAINING THE MODEL
ann.fit(x_train, y_train, batch_size=32, epochs=100)

#* PREDICTING FOR A SPECIFIC DATA POINT
y_pred_single = ann.predict(
    sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
)
print("Predicted output:", y_pred_single)

#* PREDICTING FOR TEST SET
y_pred = ann.predict(x_test)

#* COMPARING PREDICTED AND ACTUAL VALUES
print(
    np.concatenate(
        (y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)),
        axis=1
    )
)
