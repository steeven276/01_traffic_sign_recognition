
import tensorflow as tf
import numpy as np

def predict_with_model(model,img_path):
    image=tf.io.read_file(img_path)
    image=tf.image.decode_png(image, channels=3)
    image=tf.image.convert_image_dtype(image, dtype=tf.float32)
    image=tf.image.resize(image, [60,60])
    image=tf.expand_dims(image, axis=0)

    predictions=model.predict(image)

    predictions=np.argmax(predictions)

    return predictions



if __name__=="__main__":
    img_path="C:\\Users\\kenny\\01_projet_personnel\\01_projects\\00_computer_vision\\01_traffic_sign_recognition\\archive\\Test\\19\\02584.png"
    model=tf.keras.models.load_model('./Models')
    prediction=predict_with_model(model, img_path)
    print(f"prediction={prediction}")
