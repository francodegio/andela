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
Important: to be able to train the model, you first need to add a `data` folder with
the `journeys.csv` file.
```sh
cd andela
python train.py
```

### Serving the model
```sh
cd ..
uvicorn main:app --reload
```
This will serve the app locally at http://127.0.0.1/
### Testing the endpoint
Now you can send a payload to get a prediction using something like cURL, Postman, etc.
Here's a payload example to use with POST method:
```json
{
    "city_name": "San Francisco",
    "journey_starting_datetime": "2017-06-01 22:00:00",
    "journey_duration_hours": 4.4
}
```

