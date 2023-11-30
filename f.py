import cv2
import dropbox
import time
import random
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tkinter import Tk, filedialog

\
def take_snapshot():
    number = random.randint(0, 100)

    video_capture_object = cv2.VideoCapture(0)
    time.sleep(2)  # Allow the camera to warm up
    ret, frame = video_capture_object.read()

    image_name = f"img{number}.png"
    cv2.imwrite(image_name, frame)

    video_capture_object.release()
    cv2.destroyAllWindows()

    return image_name


def upload_file(image_name):
    access_token = "your_dropbox_access_token"
    file_from = image_name
    file_to = f"/Automation/{image_name}"
    
    dbx = dropbox.Dropbox(access_token)

    with open(file_from, 'rb') as f:
        dbx.files_upload(f.read(), file_to, mode=dropbox.files.WriteMode.overwrite)
        print("File Uploaded")


def perform_regression(X, y):
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')

    
    model.fit(X_train_scaled, y_train, epochs=50, batch_size=32, validation_split=0.2, verbose=0)

  
    mse = model.evaluate(X_test_scaled, y_test)
    print(f'Mean Squared Error: {mse}')

  
    predictions = model.predict(X_test_scaled)
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, predictions.flatten())
    plt.title('Actual vs. Predicted Solar Gravity')
    plt.xlabel('Actual Solar Gravity')
    plt.ylabel('Predicted Solar Gravity')
    plt.show()

def load_data_gui():
    root = Tk()
    root.title("Data Loading GUI")

    file_path = filedialog.askopenfilename(title="Select CSV file", filetypes=[("CSV files", "*.csv")])

    if file_path:
        df = pd.read_csv(file_path)
        root.destroy()
        return df
    else:
        root.destroy()
        return None


def main():
    start_time = time.time()

    while True:
        if (time.time() - start_time) >= 60:
            # Take a snapshot and upload to Dropbox
            name = take_snapshot()
            upload_file(name)
\
            X = np.random.rand(100, 1)
            y = 3 * X.flatten() + 2 + 0.1 * np.random.randn(100)

            perform_regression(X, y)

            start_time = time.time()

if __name__ == "__main__":
    main()
