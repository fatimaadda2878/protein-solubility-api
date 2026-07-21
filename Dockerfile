FROM python:3.10-slim

LABEL maintainer="Fatima Adda-Rezig"
LABEL description="API FastAPI de prédiction de solubilité des protéines"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODEL_PATH=/app/model/lgbm_model.joblib \
    ONNX_PATH=/app/model/lgbm_model.onnx \
    MODEL_META_PATH=/app/model/model_meta.json

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY model/ ./model/

RUN python -c "\
import joblib, os; \
import onnxmltools; \
from onnxmltools.convert import convert_lightgbm; \
from onnxmltools.convert.common.data_types import FloatTensorType; \
model = joblib.load('model/lgbm_model.joblib'); \
onnx_model = convert_lightgbm(model.booster_, initial_types=[('float_input', FloatTensorType([None, 13]))], target_opset=12); \
open('model/lgbm_model.onnx', 'wb').write(onnx_model.SerializeToString()); \
print('ONNX genere dans Docker OK -', os.path.getsize('model/lgbm_model.onnx'), 'bytes') \
"

RUN mkdir -p /app/logs

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl --fail http://localhost:7860/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
