from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, concatenate

# Inputs
image_input = Input(shape=(2048,))
text_input = Input(shape=(max_length,))

# Image feature processing
image_dense = Dense(256, activation="relu")(image_input)

# Text embedding and processing
text_embedding = Embedding(vocab_size, 256)(text_input)
text_lstm = LSTM(256)(text_embedding)

# Combine features
combined = concatenate([image_dense, text_lstm])
output = Dense(vocab_size, activation="softmax")(combined)

# Define and compile model
model = Model(inputs=[image_input, text_input], outputs=output)
model.compile(optimizer="adam", loss="categorical_crossentropy")

# Train the model
model.fit(
    [np.array(list(features_dict.values())), padded_sequences],
    padded_sequences,
    epochs=10,
    batch_size=32,
)
