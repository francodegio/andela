# Andela

## MLE Challenge

Here you can find the code for training the model, exposing the API endpoint and
creating the visualizations presented in the slides.

## How to

### Installation
It is recommended to create a pyenv or conda environment for installing the package and
requirements.
```sh
git clone https://github.com/francodegio/andela.git
cd andela
pip install -e .
```

### Training the model
```sh
cd andela
python train.py
```

### Serving the model
```sh
cd ..
uvicorn main:app --reload
```