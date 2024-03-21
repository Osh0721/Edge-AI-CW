
import numpy as np
from scipy.spatial.distance import cosine
import tensorflow as tf
from PIL import Image, UnidentifiedImageError
import pandas as pd
import os


def is_image_file(filename):
    valid_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']
    return any(filename.lower().endswith(ext) for ext in valid_extensions)

def load_image(image_path):
    image = Image.open(image_path)
    image = image.resize((160, 160))
    image = np.asarray(image)
    image = image.astype('float32')
    mean, std = image.mean(), image.std()
    image = (image - mean) / std
    image = np.expand_dims(image, axis=0)
    return image

def load_pb_model(model_filepath):
    graph = tf.Graph()
    graph_def = tf.compat.v1.GraphDef()
    with open(model_filepath, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def, name="")
    return graph

def generate_embeddings_dataframe(graph, dataset_path):
    person_embeddings = {}
    with tf.compat.v1.Session(graph=graph) as sess:
        for person_name in os.listdir(dataset_path):
            person_path = os.path.join(dataset_path, person_name)
            if os.path.isdir(person_path):
                person_embeddings[person_name] = []
                for image_name in os.listdir(person_path):
                    if not is_image_file(image_name):
                        continue
                    image_path = os.path.join(person_path, image_name)
                    try:
                        image = load_image(image_path)
                    except UnidentifiedImageError:
                        print(f"Skipping file (not an image): {image_path}")
                        continue
                    images_placeholder = graph.get_tensor_by_name("input:0")
                    embeddings_tensor = graph.get_tensor_by_name("embeddings:0")
                    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
                    feed_dict = {images_placeholder: image, phase_train_placeholder: False}
                    embedding = sess.run(embeddings_tensor, feed_dict=feed_dict).flatten()
                    person_embeddings[person_name].append(embedding)
    df = pd.DataFrame(list(person_embeddings.items()), columns=['Person', 'Embeddings'])
    return df

def predict_with_similarity_scores(graph, image_path, embeddings_df, threshold=0.5):
    new_embedding = get_embedding(graph, image_path)
    similarity_scores = {}
    for _, row in embeddings_df.iterrows():
        person = row['Person']
        person_scores = []
        for embedding in row['Embeddings']:
            distance = cosine(new_embedding, embedding)
            similarity = 1 - distance
            person_scores.append(similarity)
        average_similarity = np.mean(person_scores)
        similarity_scores[person] = average_similarity
    best_match = max(similarity_scores, key=similarity_scores.get)
    best_similarity_score = similarity_scores[best_match]
    if best_similarity_score < threshold:
        return "Not in database", None, similarity_scores
    return best_match, best_similarity_score, similarity_scores

def get_embedding(graph, image_path):
    image = load_image(image_path)
    with tf.compat.v1.Session(graph=graph) as sess:
        images_placeholder = graph.get_tensor_by_name("input:0")
        embeddings_tensor = graph.get_tensor_by_name("embeddings:0")
        phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
        feed_dict = {images_placeholder: image, phase_train_placeholder: False}
        embedding = sess.run(embeddings_tensor, feed_dict=feed_dict)
    return embedding.flatten()

model_filepath = '20180402-114759/20180402-114759.pb'
graph = load_pb_model(model_filepath)
dataset_path = 'database'
embeddings_df = generate_embeddings_dataframe(graph, dataset_path)

test_image_path = 'test/person/Photo on 2024-03-12 at 2.21 PM.jpg'
predicted_person, best_similarity, all_similarity_scores = predict_with_similarity_scores(graph, test_image_path, embeddings_df)

if predicted_person != "Not in database":
    print(f"Predicted person: {predicted_person}, Best Similarity: {best_similarity:.2f}")
    print("Similarity Scores with all individuals:")
    for person, score in all_similarity_scores.items():
        print(f"{person}: {score:.2f}")
else:
    print(predicted_person)