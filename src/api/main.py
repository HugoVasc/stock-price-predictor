from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
import uvicorn
from mangum import Mangum

# Carregar modelo PyTorch
MODEL_PATH = "lstm_stock_data_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

# Definir a estrutura dos dados de entrada
class StockFeatures(BaseModel):
    features: list[float]

app = FastAPI()

@app.post("/predict")
def predict(stock_data: StockFeatures):
    try:
        # Converter entrada para tensor
        input_tensor = torch.tensor([stock_data.features], dtype=torch.float32).to(device)
        
        # Fazer a previsão
        with torch.no_grad():
            prediction = model(input_tensor).cpu().numpy()
        
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

handler = Mangum(app)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
