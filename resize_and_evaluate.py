import numpy as np
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


image_sizes = [4, 6, 8]

for size in image_sizes:
    # Resize images to the specified size
    resized_images = [resize(image, (size, size)) for image in images]

    # Split the data into train, dev, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(
        resized_images, labels, train_size=0.7, random_state=42)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42)

    train_acc = accuracy_score(y_train, model.predict(X_train))
    dev_acc = accuracy_score(y_dev, model.predict(X_dev))
    test_acc = accuracy_score(y_test, model.predict(X_test))

    # Print the results
    print(f"Image size: {size}x{size} Train Acc: {train_acc:.2f} Dev Acc: {dev_acc:.2f} Test Acc: {test_acc:.2f}")
