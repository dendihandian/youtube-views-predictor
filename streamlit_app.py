from utilities.helper import min_max, model_names, predict, metrics_result
import streamlit as st

###### Variables
likes = 0
dislikes = 0
comment_count = 0
views = None

###### The App

st.title('Youtube Views Predictor', anchor=None)

st.write("""
    - [Exploratory Data Analysis](https://github.com/dendihandian/youtube-views-predictor/blob/main/trending-youtube-video-statistics.ipynb)
    - [Dataset](https://www.kaggle.com/datasnaek/youtube-new)
""")

st.write("""__________""")

with st.container():

    st.header('Views Predictor')

    with st.container():

        likes = st.slider('Likes count', min_value=min_max['likes'][0], max_value=min_max['likes'][1], value=likes)
        dislikes = st.slider('Dislikes count', min_value=min_max['dislikes'][0], max_value=min_max['dislikes'][1], value=dislikes)
        comment_count = st.slider('Comment count', min_value=min_max['comment_count'][0], max_value=min_max['comment_count'][1], value=comment_count)

        modelname = st.selectbox('Model', model_names)

        if st.button('Predict'):
            views = predict({
                'likes': likes,
                'dislikes': dislikes,
                'comment_count': comment_count,
            }, modelname)

        if views != None:
            st.metric('Views', round(views), delta=None, delta_color="normal")

st.write("""__________""")

with st.container():
    st.header('Model Evaluation')
    st.dataframe(metrics_result)