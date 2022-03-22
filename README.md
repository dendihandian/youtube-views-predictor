# Youtube Views Predictor

Youtube Views Predictor based on `Trending Youtube Video Statistic` dataset

## Requirement

Development:
- Anaconda
- Streamlit Package (Install via Anaconda)

Production (Streamlit):
- Python 3.9 or newer
- Poetry Dependency Manager

## Development

It's better to create a new Anaconda Environment named `streamlit_related` or `streamlit_projects` or anything. You can use conda command to create it:

```
conda create -n streamlit_related python=3.8
```

use the new created env:

```
conda activate streamlit_related
```

install streamlit package:
```
pip install streamlit
```
```
pip install sklearn
```

run the streamlit development server:
```
streamlit run streamlit_app.py
```

## Deployment (Streamlit)

> `streamlit` package already added in the streamlit runtime. no need to install.

install production dependencies:
```
poetry install
```

export production dependencies:
```
poetry export -f requirements.txt --without-hashes > requirements.txt
```
