"""Script de génération du modèle ONNX — exécuté lors du build Docker."""
import joblib
import os
from onnxmltools.convert import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType

model = joblib.load("model/lgbm_model.joblib")
initial_type = [("float_input", FloatTensorType([None, 13]))]
onnx_model = convert_lightgbm(
    model.booster_,
    initial_types=initial_type,
    target_opset=12
)
with open("model/lgbm_model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())

size = os.path.getsize("model/lgbm_model.onnx")
print(f"ONNX genere dans Docker OK - {size} bytes")
