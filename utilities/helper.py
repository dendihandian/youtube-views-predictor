import pickle
import pandas as pd
import numpy as np

model_names = (
    'Linear Regression',
    'Ridge (Hyperparameter Tuned)',
    'Lasso (Hyperparameter Tuned)',
    'ElasticNet (Hyperparameter Tuned)',
    'Decision Tree Regressor',
)

min_max = pickle.load(open('serialization/utilities/min_and_max.pickle', 'rb')) # NOTE: you can debug the min_max using st.json(min_max)
scaler_likes = pickle.load(open('serialization/utilities/scaler_likes.pickle', 'rb'))
scaler_dislikes = pickle.load(open('serialization/utilities/scaler_dislikes.pickle', 'rb'))
scaler_comment_count = pickle.load(open('serialization/utilities/scaler_comment_count.pickle', 'rb'))
scaler_views = pickle.load(open('serialization/utilities/scaler_views.pickle', 'rb'))

linear_regression = pickle.load(open('serialization/models/linear_regression.pickle', 'rb'))
decision_tree_regressor = pickle.load(open('serialization/models/decision_tree_regressor.pickle', 'rb'))
elasticnet_tuned = pickle.load(open('serialization/models/elasticnet_tuned.pickle', 'rb'))
lasso_tuned = pickle.load(open('serialization/models/lasso_tuned.pickle', 'rb'))
ridge_tuned = pickle.load(open('serialization/models/ridge_tuned.pickle', 'rb'))

###### Serialized Dataframes
metrics_result = pickle.load(open('serialization/dataframes/metrics_result.pickle', 'rb'))

def predict(parameters, modelname):

    likes = parameters['likes']
    dislikes = parameters['dislikes']
    comment_count = parameters['comment_count']

    inputs = pd.DataFrame({
        'likes': [scaler_likes.transform(np.array([likes]).reshape(1, -1))[0][0]],
        'dislikes': [scaler_dislikes.transform(np.array([dislikes]).reshape(1, -1))[0][0]],
        'comment_count': [scaler_comment_count.transform(np.array([comment_count]).reshape(1, -1))[0][0]]
    })

    output = None
    views = 0

    if (modelname == model_names[0]):

        output = linear_regression.predict(inputs)
        output = output[0][0]

    elif (modelname == model_names[1]):

        output = ridge_tuned.predict(inputs)
        output = output[0][0]

    elif (modelname == model_names[2]):

        output = lasso_tuned.predict(inputs)
        output = output[0]

    elif (modelname == model_names[3]):

        output = elasticnet_tuned.predict(inputs)
        output = output[0]

    elif (modelname == model_names[4]):

        output = decision_tree_regressor.predict(inputs)
        output = output[0]

    # NOTE: some output from the model may have different shape. debug it with st.write(output.shape)

    if (output):

        views = scaler_views.inverse_transform(np.array([output]).reshape(1, -1))[0][0]

    return views