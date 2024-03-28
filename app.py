#training neural network model
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dot, Dense
from tensorflow.keras.optimizers import Adam

# Define model parameters
num_users = 1000  
num_movies = 1000 
embedding_size = 50

# Model definition
user_input = Input(shape=(1,), name='user_input')
user_embedding = Embedding(num_users, embedding_size, name='user_embedding')(user_input)
user_vec = Flatten(name='user_flatten')(user_embedding)

movie_input = Input(shape=(1,), name='movie_input')
movie_embedding = Embedding(num_movies, embedding_size, name='movie_embedding')(movie_input)
movie_vec = Flatten(name='movie_flatten')(movie_embedding)

dot_product = Dot(axes=1)([user_vec, movie_vec])
output = Dense(1, activation='sigmoid')(dot_product)

model = Model(inputs=[user_input, movie_input], outputs=output)
model.compile(optimizer=Adam(0.001), loss='binary_crossentropy')

# Assume X_user, X_movie, and y as your input data and labels
model.fit([X_user, X_movie], y, epochs=5, batch_size=32)


import tf2onnx
import tensorflow as tf

# Specify the inputs & outputs for the model
spec = (tf.TensorSpec((None, 1), tf.int32, name="user_input"),
        tf.TensorSpec((None, 1), tf.int32, name="movie_input"))

# Convert the TensorFlow model to ONNX
output_path = "recommendation_model.onnx"
model_proto, _ = tf2onnx.convert.from_keras(model, input_signature=spec, output_path=output_path)


import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path):
    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(1) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = 1 << 30  # 1GB
        builder.max_batch_size = 1
        with open(onnx_file_path, 'rb') as model:
            parser.parse(model.read())
        return builder.build_cuda_engine(network)

engine = build_engine('recommendation_model.onnx')



