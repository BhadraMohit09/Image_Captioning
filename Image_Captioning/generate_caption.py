def generate_caption(image_path, model, tokenizer, max_length):
    features = extract_features(image_path)
    input_text = "<start>"
    
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([input_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length, padding="post")
        
        # Predict the next word
        predicted_id = model.predict([features, sequence]).argmax()
        predicted_word = tokenizer.index_word.get(predicted_id, "")
        
        if predicted_word == "<end>":
            break
        
        input_text += " " + predicted_word

    return input_text.replace("<start>", "").strip()

# Example usage
print(generate_caption("test_image.jpg", model, tokenizer, max_length))
