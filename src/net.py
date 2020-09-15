import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.metrics import Recall, SpecificityAtSensitivity
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import max_norm
from sklearn.metrics import recall_score, f1_score
from model_class import Model
from data_class import Data, create_data


def create_model(optimizer='Adagrad', init_mode='zero', activation1='softsign',
                 activation2='softsign', dropout_rate=0.5, weight_constraint=3,
                 neurons1=15, neurons2=50):
    nn_model = Sequential()
    nn_model.add(Dense(12, input_dim=28, activation=activation1,
                       kernel_initializer=init_mode,
                       kernel_constraint=max_norm(weight_constraint)))
    nn_model.add(Dropout(dropout_rate))
    nn_model.add(Dense(8, activation=activation2))
    nn_model.add(Dense(1, activation='sigmoid'))
    nn_model.compile(loss='binary_crossentropy', optimizer=optimizer,
                     metrics=[Recall()])
    return nn_model


def f1_eval(y_pred, dtrain):
    # Calculate f1_score for neural net tuning
    y_true = dtrain.get_label()
    err = 1-f1_score(y_true, np.round(y_pred))
    return 'f1_err', err


if __name__ == '__main__':
    all_df = create_data('../data/merged_df.csv', 'All')

    # Create train/test sets
    X, y = Data.pop_reported(all_df.full_df)
    all_df.train_test_split(test_size=0.33)
    X_train, y_train = all_df.pop_reported(all_df.train_df)
    X_test, y_test = all_df.pop_reported(all_df.test_df)

    # Neural Net
    nn_model = KerasClassifier(build_fn=create_model, batch_size=300,
                               epochs=15)
    distributions_nn = dict(
        batch_size=[100, 200, 300, 400, 500],
        epochs=[5, 10, 15, 20, 25, 50],
        optimizer=['SGD', 'RMSprop', 'Adagrad',
                   'Adadelta', 'Adam', 'Adamax',
                   'Nadam'],
        # learn_rate=[0.000001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3],
        # momentum=[0.0, 0.2, 0.4, 0.6, 0.8, 0.9],
        init_mode=['uniform', 'lecun_uniform', 'normal',
                   'zero', 'glorot_normal',
                   'glorot_uniform', 'he_normal',
                   'he_uniform'],
        activation1=['softmax', 'softplus', 'softsign',
                     'relu', 'tanh', 'sigmoid',
                     'hard_sigmoid', 'linear'],
        activation2=['softmax', 'softplus', 'softsign',
                     'relu', 'tanh', 'sigmoid',
                     'hard_sigmoid', 'linear'],
        weight_constraint=[2, 3, 4],
        dropout_rate=[0.4, 0.5, 0.6, 0.7],
        neurons1=[1, 5, 10, 15, 25, 50],
        neurons2=[1, 5, 10, 15, 25, 50]
    )
    NN = Model(nn_model, f1_score)
    print(NN.hyper_search(distributions_nn, X_train, y_train))
    # ({'neurons1': 15, 'activation2': 'linear', 'weight_constraint': 2,
    # 'neurons2': 25, 'dropout_rate': 0.5, 'optimizer': 'Adamax',
    # 'init_mode': 'glorot_uniform', 'activation1': 'softplus',
    # 'batch_size': 100, 'epochs': 25}, 0.3632652246007728)
    print(NN.hyper_search(distributions_nn, X_train, y_train))
    # R1b_matrix = NN.confusion_matrix(X_test, y_test.values)
    # ({'weight_constraint': 4, 'dropout_rate': 0.5, 'activation2': 'relu',
    # 'batch_size': 200, 'activation1': 'relu', 'optimizer': 'RMSprop',
    # 'epochs': 50, 'neurons2': 5, 'init_mode': 'he_normal',
    # 'neurons1': 1}, 0.4642209424562366)

    # Change to over sampling and repeat
    X_train, y_train = all_df.over_sampling(all_df.train_df)

    print(NN.hyper_search(distributions_nn, X_train, y_train))
    # ({'init_mode': 'glorot_uniform', 'batch_size': 100,
    # 'optimizer': 'Adamax', 'dropout_rate': 0.5, 'weight_constraint': 2,
    # 'neurons2': 25, 'activation1': 'softplus', 'activation2': 'linear',
    # 'epochs': 25, 'neurons1': 15}, 0.636014040161523)
    print(NN.hyper_search(distributions_nn, X_train, y_train))
    # ({'dropout_rate': 0.5, 'init_mode': 'zero', 'activation1': 'softsign',
    # 'weight_constraint': 3, 'neurons1': 15, 'optimizer': 'Adagrad',
    # 'batch_size': 100, 'epochs': 50, 'neurons2': 50,
    # 'activation2': 'softsign'}, 0.655386722306166)
