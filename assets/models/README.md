# TFLite model for image similarity (tflite_flutter 0.12.1)

This app loads the model **via the tflite_flutter package API** (`Interpreter.fromAsset`). It first tries the package asset path, then this folder.

**Included:** `mobilenet_quant.tflite` — same model as in the tflite_flutter package example (image classification). It is used to get a 1001-dim vector per image; similarity is the cosine similarity of these vectors.

**Model:** MobileNet quant, input `[1, 224, 224, 3]` uint8, output `[1, 1001]`.
