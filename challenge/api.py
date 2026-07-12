from typing import List

import fastapi
import pandas as pd
from pydantic import BaseModel

from challenge.model import ReplenishmentModel

app = fastapi.FastAPI()


class ProductRequest(BaseModel):
    gtin: str
    fecha: str


class PredictRequest(BaseModel):
    products: List[ProductRequest]


# Datos base del challenge.
_MOVIMIENTOS = pd.read_csv("dataset/movimientos.csv")
_PRODUCTOS = pd.read_csv("dataset/productos.csv")
_VALID_GTINS = set(_PRODUCTOS["gtin"].astype(str))

# Modelo entrenado al iniciar la API.
_MODEL = ReplenishmentModel()
_FEATURES, _TARGET = _MODEL.preprocess(
    data=_MOVIMIENTOS,
    target_column="cantidad"
)
_MODEL.fit(
    features=_FEATURES,
    target=_TARGET
)


@app.get("/health", status_code=200)
async def get_health() -> dict:
    return {
        "status": "OK"
    }


@app.post("/predict", status_code=200)
async def post_predict(request: PredictRequest) -> dict:
    rows = []

    for product in request.products:
        gtin = str(product.gtin)

        if gtin not in _VALID_GTINS:
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"Producto desconocido: {gtin}"
            )

        fecha = pd.to_datetime(product.fecha, errors="coerce")

        if pd.isna(fecha):
            raise fastapi.HTTPException(
                status_code=400,
                detail=f"Fecha inválida: {product.fecha}"
            )

        rows.append(
            {
                "gtin": gtin,
                "fecha": fecha.strftime("%Y-%m-%d"),
                "tipo_movimiento": "S"
            }
        )

    input_data = pd.DataFrame(rows)
    features = _MODEL.preprocess(data=input_data)
    predictions = _MODEL.predict(features=features)

    return {
        "predict": predictions
    }