import argparse
import os
import tensorflow as tf
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description="Debugging script for segmentation fault.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--dataset_path", type=str, default=os.getcwd()+"/training_data")
    parser.add_argument("--image_width", type=int, default=324)
    parser.add_argument("--image_height", type=int, default=244)
    parser.add_argument("--image_channels", type=int, default=1)
    return parser.parse_args()

def check_images(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            print(f"Checking file: {os.path.join(root, file)}")
            try:
                img = Image.open(os.path.join(root, file))
                img.verify()
            except (IOError, SyntaxError) as e:
                print(f"Corrupted image file: {os.path.join(root, file)} - {e}")

if __name__ == "__main__":
    args = parse_args()
    
    
    # Enable TensorFlow debugging logs
    # tf.debugging.set_log_device_placement(True)
    # tf.get_logger().setLevel('INFO')
    
    print(f"Checking dataset...{args.dataset_path}")
    # Check for corrupted images
    check_images(args.dataset_path)
    
    # Setup data generators
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    train_generator = train_datagen.flow_from_directory(
        os.path.join(args.dataset_path, "train"),
        target_size=(args.image_width, args.image_height),
        batch_size=args.batch_size,
        class_mode="categorical",
        color_mode="grayscale"
    )

    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator()
    val_generator = val_datagen.flow_from_directory(
        os.path.join(args.dataset_path, "validation"),
        target_size=(args.image_width, args.image_height),
        batch_size=args.batch_size,
        class_mode="categorical",
        color_mode="grayscale"
    )
    
    # Simple model for debugging
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(args.image_width, args.image_height, args.image_channels)),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    model.summary()
    
    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=args.epochs,
        validation_data=val_generator,
        validation_steps=len(val_generator)
    )
    
    print("Training completed successfully")
