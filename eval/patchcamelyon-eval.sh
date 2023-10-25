#!/bin/bash

MODEL_DIR="saved_model_dir"
VENV_PATH="caval/bin/activate"  # default path
REQUIREMENTS=""
BATCH_SIZE=128  # default batch size
DATASET_DIR="~/tensorflow_datasets/patch_camelyon"  # default tfds storage path

# parse command line arguments
while getopts ":m:v:r:b:" opt; do
  case $opt in
    m) MODEL_DIR="$OPTARG"
    ;;
    v) VENV_PATH="$OPTARG"
    ;;
    r) REQUIREMENTS="$OPTARG"
    ;;
    b) BATCH_SIZE="$OPTARG"
    ;;
    \?) echo "invalid option -$OPTARG" >&2
    ;;
  esac
done

# step 1: check if virtual environment exists, create if not
if [ ! -d "${VENV_PATH%/*}" ]; then
  if [ -z "$REQUIREMENTS" ]; then
    echo "virtual environment does not exist and no requirements.txt provided. exiting."
    exit 1
  else
    echo "creating virtual environment..."
    python3 -m venv ${VENV_PATH%/*}
    source $VENV_PATH
    pip install -r $REQUIREMENTS
  fi
else
  # activate python virtual environment
  source $VENV_PATH
fi

# step 2 and 3: download patchcamelyon test data (if not already) and evaluate the model
python << END
import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import os
import numpy
import sklearn

# Check if dataset is already downloaded
if not os.path.exists("$DATASET_DIR"):
    # downloading the patchcamelyon dataset
    tfds.load('patch_camelyon', split='test', download=True)

@tf.autograph.experimental.do_not_convert
def preprocess(image, label):
    """Converts image dtype to float, normalizes, and then scales it back to [0, 255] range."""
    image = tf.image.convert_image_dtype(image, tf.float32)  # Convert dtype to float and normalize to [0, 1]
    image = image * 255  # Scale back to [0, 255]
    return image, label

for split in ('train', 'test', 'validation'):    
    # Loading the patchcamelyon dataset
    patchcamelyon_data, _ = tfds.load('patch_camelyon', split=split, with_info=True, download=False, as_supervised=True)
    
    # Apply preprocessing
    patchcamelyon_data = patchcamelyon_data.map(preprocess)
    
    def create_placeholder_model():
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print("gpu(s) available. tensorflow will use gpu:", gpus[0])
    else:
        print("no gpu available. tensorflow will use cpu.")
    
    all_predictions = []
    all_labels = []
    
    if tf.io.gfile.exists("$MODEL_DIR"):
        model = tf.saved_model.load("$MODEL_DIR")
        infer = model.signatures["pred_fn"]
        for images, labels in patchcamelyon_data.batch($BATCH_SIZE):
            batch_predictions = infer(tf.constant(images))['output_0'].numpy()
            #print("images")
            #print(images)
            #print("raw batch")
            #print(batch_predictions)
            predicted_class = batch_predictions.argmax(axis=1)
            all_predictions.extend(predicted_class)
            #print("preds")
            #print(predicted_class)
            all_labels.extend(labels.numpy())
            #print("gt")
            #print(labels)
    else:
        print("using placeholder model as no saved model was provided.")
        model = create_placeholder_model()
        for images, labels in patchcamelyon_data.batch($BATCH_SIZE):
            batch_predictions = model.predict(images)
            predicted_class = batch_predictions.argmax(axis=1)
            all_predictions.extend(predicted_class)
            all_labels.extend(labels.numpy())
    
    #print(numpy.sum(all_labels))
    #print(numpy.sum(all_predictions))
    
    #f1 = f1_score(all_labels, all_predictions)
    #print(f"f1 score: {f1}")
    
    #cm = confusion_matrix(all_labels,all_predictions)
    
    #print(cm)

    print(split)
    ba = sklearn.metrics.balanced_accuracy_score(all_labels,all_predictions)
    print("ba: ", ba)
    
    
    #report = classification_report(all_labels, all_predictions)
    #print(report)
END
