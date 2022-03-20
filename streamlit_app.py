import streamlit as st
import pandas as pd
import numpy as np
import pickle

###### Load Models
linear_regression = pickle.load(open('serialization/models/linear_regression.pickle', 'rb'))
decision_tree_regressor = pickle.load(open('serialization/models/decision_tree_regressor.pickle', 'rb'))
elasticnet_tuned = pickle.load(open('serialization/models/elasticnet_tuned.pickle', 'rb'))
lasso_tuned = pickle.load(open('serialization/models/lasso_tuned.pickle', 'rb'))
ridge_tuned = pickle.load(open('serialization/models/ridge_tuned.pickle', 'rb'))

###### Load Utilities
min_max = pickle.load(open('serialization/utilities/min_and_max.pickle', 'rb')) # use st.json(min_max) to debug
scaler_likes = pickle.load(open('serialization/utilities/scaler_likes.pickle', 'rb'))
scaler_dislikes = pickle.load(open('serialization/utilities/scaler_dislikes.pickle', 'rb'))
scaler_comment_count = pickle.load(open('serialization/utilities/scaler_comment_count.pickle', 'rb'))
scaler_views = pickle.load(open('serialization/utilities/scaler_views.pickle', 'rb'))

###### Variables
likes = 0
dislikes = 0
comment_count = 0
views = None
predicting = False
models = (
    'Linear Regression',
    'Ridge (Hyperparameter Tuned)',
    'Lasso (Hyperparameter Tuned)',
    'ElasticNet (Hyperparameter Tuned)',
    'Decision Tree Regressor',
)

###### Functions

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

    if (modelname == models[0]):

        output = linear_regression.predict(inputs)

    elif (modelname == models[1]):

        output = ridge_tuned.predict(inputs)

    elif (modelname == models[2]):

        output = lasso_tuned.predict(inputs)

    elif (modelname == models[3]):

        output = elasticnet_tuned.predict(inputs)

    elif (modelname == models[4]):

        output = decision_tree_regressor.predict(inputs)

    if (output):
        views = scaler_views.inverse_transform(np.array([output[0][0]]).reshape(1, -1))[0][0]

    return views

###### The App

st.title('Youtube Views Predictor', anchor=None)

with st.container():

    st.header('Views Regressor')

    with st.container():

        likes = st.slider('Likes count', min_value=min_max['likes'][0], max_value=min_max['likes'][1], value=likes)
        dislikes = st.slider('Dislikes count', min_value=min_max['dislikes'][0], max_value=min_max['dislikes'][1], value=dislikes)
        comment_count = st.slider('Comment count', min_value=min_max['comment_count'][0], max_value=min_max['comment_count'][1], value=comment_count)

        modelname = st.selectbox('Model', models)

        predicting = st.button('Predict')

        if predicting:
            views = predict({
                'likes': likes,
                'dislikes': dislikes,
                'comment_count': comment_count,
            }, modelname)

        if views != None:
            st.metric('Views', round(views), delta=None, delta_color="normal")


# with st.container():
#     st.header('TODO')