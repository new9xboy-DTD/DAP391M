from data_augmentation import data_augmentation
import os

train_datagen, val_datagen, test_datagen = data_augmentation()

# Define dataset paths
dataset_root = os.path.join("Dataset")
train_dir = os.path.join(dataset_root, "Train")
val_dir = os.path.join(dataset_root, "Validation")
test_dir = os.path.join(dataset_root, "Test")

# Image dimensions and batch size
img_width, img_height = 256, 256
batch_size = 32

def data_preprocessing():
  train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
    )

  val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
  )

  test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
  )

  return train_generator, val_generator, test_generator