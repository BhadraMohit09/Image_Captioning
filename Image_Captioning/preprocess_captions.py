tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)

# Convert captions to sequences
sequences = tokenizer.texts_to_sequences(captions)

# Pad sequences
max_length = max(len(seq) for seq in sequences)
padded_sequences = pad_sequences(sequences, maxlen=max_length, padding="post")

# Vocabulary size
vocab_size = len(tokenizer.word_index) + 1
