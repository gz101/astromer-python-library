import json
import os

import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Lambda,
)

from ASTROMER.core.astromer import get_ASTROMER
from ASTROMER.core.output import RegLayer


def _build_input_placeholder(window_size: int = 200) -> dict[str, Input]:
    series = Input(
        shape=(window_size, 1), batch_size=None, name="input"
    )
    times = Input(
        shape=(window_size, 1), batch_size=None, name="times"
    )
    mask = Input(
        shape=(window_size, 1), batch_size=None, name="mask_in"
    )
    length = Input(
        shape=(window_size, 1), batch_size=None, name="length"
    )

    return {
        "input": series,
        "mask_in": mask,
        "times": times,
        "length": length,
    }


def get_MBAstromer(model_names: list[str], collapse="average") -> Model:
    input_placeholder = _build_input_placeholder()
    outputs = []

    for i, model_name in enumerate(model_names):
        local_path = os.path.join("./weights", model_name)

        if not os.path.isdir(local_path):
            raise ValueError(f"Pre-trained model not found: {local_path}.")

        # config file
        conf_file = os.path.join(local_path, "conf.json")
        with open(conf_file, "r") as handle:
            conf = json.load(handle)

        model = get_ASTROMER(
            num_layers=conf["layers"],
            d_model=conf["head_dim"],
            num_heads=conf["heads"],
            dff=conf["dff"],
            base=conf["base"],
            dropout=conf["dropout"],
            maxlen=conf["max_obs"],
        )
        model.load_weights(os.path.join(local_path, "weights"))

        encoder = model.get_layer("encoder")
        encoder._name = f"encoder_{i}"
        encoder.trainable = False

        attention_vectors = encoder(input_placeholder)
        outputs.append(attention_vectors)

    if collapse == "average":
        x = Lambda(
            lambda x: tf.reduce_mean(x, axis=0),
            name="reduce_mean",
        )(tf.stack(outputs))
    else:
        raise ValueError("Invalid `collapse` method.")

    x = RegLayer(name="regression")(x)
    return Model(inputs=input_placeholder, outputs=x, name="Multi-Band ASTROMER")
