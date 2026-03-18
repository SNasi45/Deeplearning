import numpy as np
import cv2
import os

# ----- Step 1: Set Path (GitHub Friendly) -----
base_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(base_dir, "..", "img")
path = os.path.abspath(path)

print("Using image folder:", path)

# ----- Step 2: Read Images -----
img_list = []

for file in os.listdir(path):
    full_path = os.path.join(path, file)

    # skip non-image files (important)
    if not file.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print("Error loading:", file)
        continue

    img = cv2.resize(img, (64, 64))
    img = img.flatten()

    img_list.append(img)

# ----- Step 3: Convert to Matrix X -----
X = np.array(img_list)
print("Shape of X:", X.shape)

# ensure exactly 10 images
if X.shape[0] != 10:
    raise ValueError("Please ensure exactly 10 valid images in img folder")

# ----- Step 4: Mean Centering -----
mean = np.mean(X, axis=0)
Xc = X - mean

# ----- Step 5: Covariance Matrix -----
cov = np.cov(Xc, rowvar=False)

# ----- Step 6: Eigen Decomposition -----
eig_val, eig_vec = np.linalg.eigh(cov)

# sort eigenvalues in descending order
idx = np.argsort(eig_val)[::-1]
eig_vec = eig_vec[:, idx]

# ----- Step 7: Decoder Matrix D -----
D = eig_vec[:, :30]
print("Shape of D:", D.shape)

# ----- Step 8: Encoded Matrix C -----
C = np.dot(Xc, D)
print("Shape of C:", C.shape)

# ----- Done -----
print("\nPCA computation completed successfully")