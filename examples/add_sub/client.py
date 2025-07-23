from tritonclient.utils import *
import tritonclient.http as httpclient
import numpy as np

triton_url = "localhost:8000"
model_name = "openpom_model"

smiles = ["CC(=O)OC1=CC=CC=C1C(=O)O"]
input_data = np.array([s.encode("utf-8") for s in smiles], dtype=np.bytes_).reshape(len(smiles), 1)

with httpclient.InferenceServerClient(triton_url) as client:
    inputs = [
        httpclient.InferInput(
            name="SMILES",
            shape=input_data.shape,
            datatype="BYTES"
        )
    ]
    inputs[0].set_data_from_numpy(input_data)
    print("Input dtype:", input_data.dtype)  # 
    print("Input shape:", input_data.shape)  #(1, 1)
    print("Input content:", input_data)
    print("input_data[0, 0]:", input_data[0, 0], type(input_data[0, 0]))


    outputs = [
        httpclient.InferRequestedOutput("OUTPUT", binary_data=True)
    ]
    print("Type of outputs[0]:", type(outputs[0]))
    print("Outputs:", outputs)

    response = client.infer(
        model_name=model_name,
        inputs=inputs,
        outputs=outputs
    )

    embeddings = response.as_numpy("OUTPUT")
    print("shape:", embeddings.shape)
    print("embedding vector: \n", embeddings[0])
