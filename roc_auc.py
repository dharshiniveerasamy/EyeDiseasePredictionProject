import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc, roc_auc_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Step 1: Set file paths
MODEL_PATH = "model/eye_disease_model.keras"
TEST_DIR = "dataset/test"
OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Step 2: Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Step 3: Load test images
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(
    rescale=1.0 / 255
)

test_data = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

# Step 4: Get actual labels and predictions
y_true = test_data.classes
y_score = model.predict(test_data, verbose=1)

class_names = list(test_data.class_indices.keys())
n_classes = len(class_names)

print("\nDisease classes:", class_names)

# Step 5: Check model output
if y_score.shape[1] != n_classes:
    raise ValueError(
        "Model output does not match the number of test classes."
    )

# Step 6: Convert labels to one-vs-rest format
y_true_binary = label_binarize(
    y_true,
    classes=np.arange(n_classes)
)

# Step 7: Calculate and draw ROC curves
plt.figure(figsize=(9, 7))

for i, class_name in enumerate(class_names):

    fpr, tpr, thresholds = roc_curve(
        y_true_binary[:, i],
        y_score[:, i]
    )

    auc_score = auc(fpr, tpr)

    plt.plot(
        fpr,
        tpr,
        label=f"{class_name} (AUC = {auc_score:.3f})"
    )

    print(f"{class_name} AUC: {auc_score:.4f}")

# Step 8: Add reference line
plt.plot(
    [0, 1],
    [0, 1],
    "k--",
    label="Random Classifier (AUC = 0.500)"
)

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Human Eye Disease Prediction")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Step 9: Save the graph
plt.savefig(
    os.path.join(OUTPUT_DIR, "roc_curve.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# Step 10: Calculate overall AUC scores
macro_auc = roc_auc_score(
    y_true,
    y_score,
    multi_class="ovr",
    average="macro"
)

weighted_auc = roc_auc_score(
    y_true,
    y_score,
    multi_class="ovr",
    average="weighted"
)

print("\nMacro-average AUC:", round(macro_auc, 4))
print("Weighted-average AUC:", round(weighted_auc, 4))
print("\nROC graph saved in results/roc_curve.png")