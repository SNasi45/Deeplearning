import numpy as np
import cv2
import os
import matplotlib.pyplot as plt

# ----- Step 1: Path -----
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "..", "img")
path = os.path.abspath(path)

print("Using image folder:", path)

# ----- Step 2: Load Images -----
img_list = []

for file in os.listdir(path):
    full_path = os.path.join(path, file)

    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error loading:", file)
        continue

    img = cv2.resize(img, (64, 64))
    img = img.flatten()

    img_list.append(img)

# ----- Step 3: Create Matrix X -----
X = np.array(img_list)
print("Shape of X:", X.shape)

# ----- Step 4: Apply SVD -----
U, S, Vt = np.linalg.svd(X, full_matrices=False)

# ----- Step 5: Print Top 5 Singular Values -----
print("\nTop 5 Singular Values:")
print(S[:5])

# ----- Step 6: Show Top 5 Singular Vectors -----
top5 = Vt[:5]

for i in range(5):
    img = top5[i].reshape(64, 64)

    # normalize image
    img = (img - img.min()) / (img.max() - img.min())

    plt.imshow(img, cmap='gray')
    plt.title(f"Singular Vector {i+1}")
    plt.axis('off')
    plt.show()

print("\nSVD Done")