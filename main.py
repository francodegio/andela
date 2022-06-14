import json

from fastapi import FastAPI

from andela.definitions import PriceModel, PricePayload


app = FastAPI()
path = 'models/price_model.pkl'
model = PriceModel(path)


@app.post('/predict')
def predict(data):
    try:
        data = PricePayload(**data)
        df = data.df
        df = model.transform(df)

        return json.dumps({"prediction": model.predict(df)}), 200
    except Exception as e:
        return json.dumps({"error": str(e)}), 400