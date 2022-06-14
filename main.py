import traceback
import logging

import pandas as pd

from fastapi import FastAPI

from andela.definitions import PriceModel, TripPayload


app = FastAPI()
logger = logging.getLogger()
try:
    path = 'models/price_model.pkl'
    model = PriceModel(path)
except:
    logger.error(traceback.format_exc())

@app.post('/')
async def predict(data: TripPayload):
    try:
        logger.info("I'm here!")
        #data = TripPayload(**data)
        df = pd.DataFrame([data.__dict__])
        df = model.transform(df)
        return {"prediction": str(model.predict(df)[0])}
    except:
        logger.error(str(traceback.format_exc()))
        return {"error":  str(traceback.format_exc())}