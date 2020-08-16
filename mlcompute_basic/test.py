import time

import numpy as np

import mlcompute as mlc


if __name__ == "__main__":
    tensor1 = mlc.tensor([6,1], "Float32")
    tensor2 = mlc.tensor([6,1], "Float32")
    tensor3 = mlc.tensor([6,1], "Float32")

    v1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).astype(np.float32)
    v2 = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]).astype(np.float32)
    v3 = np.array([-0.1, -0.1, -0.1, -0.1, -0.1, -0.1]).astype(np.float32)

    data1 = mlc.tensor_data(v1)
    data2 = mlc.tensor_data(v2)
    data3 = mlc.tensor_data(v3)

    for x in [data1, data2, data3]:
        x.print()

    g = mlc.graph()

    arith_layer1 = mlc.arithmetic_layer("add")
    arith_layer2 = mlc.arithmetic_layer("add")

    im1 = g.add_layer(arith_layer1, [tensor1, tensor2])
    result_tensor = g.add_layer(arith_layer2, [im1, tensor3])

    i = mlc.inference_graph([g])

    i.add_inputs({"data1":tensor1, "data2":tensor2, "data3":tensor3})

    device = mlc.device("gpu")

    compiling_result = i.compile(device)
    print(compiling_result)

    execute_result = i.execute({"data1":data1, "data2":data2, "data3":data3}, 0)

    time.sleep(1)
    result_tensor.print()
