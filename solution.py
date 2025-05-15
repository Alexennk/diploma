import rasterio
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import gc
file_path = "../uav/DOM_zjsru_ms-5bands_8cm.tif"

with rasterio.open(file_path) as src:
    image_array = src.read()  # shape: (bands, height, width)
    print(image_array.shape)  # (5, H, W) for a 5-band image

# RGB
rgb_array_split = np.clip(image_array[:3].copy(), 0, 1)
for i in range(3): # лучше нормализовать по отдельности
    rgb_array_split[i] = (rgb_array_split[i] - rgb_array_split[i].min()) / (rgb_array_split[i].max() - rgb_array_split[i].min())
rgb_array_split = (rgb_array_split * 255).astype(np.uint8)

# NDVI and EVI
ir_bands = np.clip(image_array[3:].copy(), 0, 2) # red edge and nir
for i in range(2):
    ir_bands[i] = (ir_bands[i] - ir_bands[i].min()) / (ir_bands[i].max() - ir_bands[i].min())
ir_bands = (ir_bands * 255).astype(np.uint8)

## NDVI
def compute_ndvi(red: np.ndarray, nir: np.ndarray):
    """
    Compute NDVI while handling division by zero.
    """
    red = red.astype(np.float32)  
    nir = nir.astype(np.float32)
    denominator = nir + red
    ndvi = np.where(denominator == 0, -1, (nir - red) / denominator)  # Assign 0 where division is undefined
    return ndvi

def ndvi_to_grayscale(ndvi):
    ndvi_normalized = ((ndvi + 1) / 2 * 255).astype(np.uint8)  
    return Image.fromarray(ndvi_normalized)
ndvi = compute_ndvi(rgb_array_split[0], ir_bands[1])

## EVI
def compute_evi(red: np.ndarray, blue: np.ndarray, nir: np.ndarray, L=1, C1=6, C2=7.5, G=2.5):
    """
    Compute EVI while handling division by zero.
    """
    red = red.astype(np.float32) 
    nir = nir.astype(np.float32)
    blue = blue.astype(np.float32)
    denominator = nir + C1 * red - C2 * blue + L
    ndvi = np.where(denominator == 0, 0, G * (nir - red) / denominator)  # Assign 0 where division is undefined
    return ndvi

def evi_to_grayscale(evi):
    evi = np.clip(evi, -1, 3)  # Clip range
    evi_scaled = ((evi - evi.min()) / (evi.max() - evi.min()) * 255).astype(np.uint8)  # Scale to [0,255]
    return Image.fromarray(evi_scaled)

evi = compute_evi(rgb_array_split[0], rgb_array_split[2], ir_bands[1])
evi_clipped = np.clip(evi, -5, 10)

# Calculate metrics
# Load labeled points
labeled_points = pd.read_csv("labeled_points.csv")

# Convert to integer coordinates for indexing
labeled_points['x'] = labeled_points['x'].astype(int)
labeled_points['y'] = labeled_points['y'].astype(int)

def compute_error_matrix(index_name, index_threshold=0.5):
    """Returns TP, TN, FP, FN"""
    index = ndvi if index_name == 'ndvi' else evi
    index_mask = index > index_threshold
    tp, tn, fp, fn = 0, 0, 0, 0
    for _, row in labeled_points.iterrows():
        x, y, label = row['x'], row['y'], row['label']
        predicted_label = 1 if index_mask[y, x] else 2
        if predicted_label == 1:
            if predicted_label == label:
                tp += 1
            else:
                fp += 1
        elif predicted_label == 2:
            if predicted_label == label:
                tn += 1
            else:
                fn += 1
    return tp, tn, fp, fn

def compute_metrics(tp, tn, fp, fn):
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    recall = tp / (tp + fn) if tp + fn != 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall !=0 else 0
    return accuracy, precision, recall, f1

def find_best_threshold(index_name='ndvi', return_other_metrics=False):
    index = ndvi if index_name == 'ndvi' else evi_clipped
    lower_bound, upper_bound = 0, index.max().astype(np.int32)
    threshold_values = [lower_bound + i * 0.01 for i in range(upper_bound * 100)]
    accuracy_values = []
    if return_other_metrics:
        precision_values = []
        recall_values = []
        f1_values = []
    for thresh in tqdm(threshold_values):
        accuracy, precision, recall, f1 = compute_metrics(*compute_error_matrix(index_name, thresh))
        accuracy_values.append(accuracy)
        if return_other_metrics:
            precision_values.append(precision)
            recall_values.append(recall)
            f1_values.append(f1)
    if return_other_metrics:
        return threshold_values, accuracy_values, precision_values, recall_values, f1_values
    return threshold_values, accuracy_values

def visualize_error_matrix(tp, tn, fp, fn):
    confusion_matrix = np.array([[tp, fn],
                                [fp, tn]])
    # Labels for axes
    labels = ['Positive', 'Negative']
    # Plotting
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels)

    plt.xlabel('Предсказанная метка')
    plt.ylabel('Истинная метка')
    plt.title('Матрица ошибок')
    plt.show()

### Best threshold for NDVI
threshold_values, accuracy_values, precision_values, recall_values, f1_values = find_best_threshold(index_name='ndvi', return_other_metrics=True)
ndvi_metrics_df = pd.DataFrame({
    'threshold': threshold_values,
    'accuracy': accuracy_values,
    'precision': precision_values,
    'recall': recall_values,
    'f1': f1_values
})
plt.plot(threshold_values, accuracy_values)
ndvi_metrics_df[ndvi_metrics_df['accuracy'] == ndvi_metrics_df['accuracy'].max()]
tp, tn, fp, fn =compute_error_matrix(index_name='ndvi', index_threshold=0.56)
accuracy, precision, recall, f1 = compute_metrics(tp, tn, fp, fn)
print(accuracy, precision, recall)
visualize_error_matrix(tp, tn, fp, fn)

### Best threshold for EVI
threshold_values, accuracy_values, precision_values, recall_values, f1_values = find_best_threshold(index_name='evi', return_other_metrics=True)
evi_metrics_df = pd.DataFrame({
    'threshold': threshold_values,
    'accuracy': accuracy_values,
    'precision': precision_values,
    'recall': recall_values,
    'f1': f1_values
})
plt.plot(threshold_values, accuracy_values)
evi_metrics_df[evi_metrics_df['accuracy'] == evi_metrics_df['accuracy'].max()]
tp, tn, fp, fn = compute_error_matrix(index_name='evi', index_threshold=1.81)
accuracy, precision, recall, f1 = compute_metrics(tp, tn, fp, fn)
print(accuracy, precision, recall)
visualize_error_matrix(tp, tn, fp, fn)

# Calculate vegetation volume
## Total area
rgb_array = image_array[:3].copy()
(rgb_array > -10000).sum() * 0.018 * 0.018

## Function for volume and area calculation 
def calculate_volume_area(height_array, index_thresh=0.56):
    ndvi_mask = ndvi > index_thresh
    veg_area = ndvi_mask.sum() * 0.018 * 0.018
    veg_volume = height_array[ndvi_mask].sum() * 0.018 * 0.018
    return veg_area, veg_volume

def calculate_co2(veg_volume):
    ro = 600
    m = ro * veg_volume # biomass
    dm = 0.725 * m # dry biomass
    cm = 0.5 * dm # carbon mass
    co2 = 3.67 * cm
    return co2

def form_volume_df(height_array):
    thresh_values = [0.4 + i * 0.05 for i in range(6)]
    co2_values = []
    area_ratios = []
    volume_ratios = []
    area_values = []
    total_volume = height_array.sum() * 0.018 * 0.018
    total_area = (height_array >= 0).sum() * 0.018 * 0.018
    for thresh in thresh_values:
        veg_area, veg_volume = calculate_volume_area(height_array, thresh)
        area_values.append(veg_area)
        volume_ratios.append(veg_volume / total_volume * 100)
        area_ratios.append(veg_area / total_area * 100)
        co2_values.append(calculate_co2(veg_volume))
    volume_df = pd.DataFrame({
        "thresh": thresh_values,
        "co2": co2_values,
        "area_ratio": area_ratios,
        "vol_ratio": volume_ratios,
        "area": area_values
    })
    return volume_df

def visualize_array_1d(input_array):
    array = input_array.copy()
    array = (array - array.min()) / (array.max() - array.min())
    array = (array * 255).astype(np.uint8)
    return Image.fromarray(array)

# visualize height map
file_path = "../uav/DSM_zjsru_op_8cm.tif"
with rasterio.open(file_path) as src:
    height_array = src.read()  # shape: (bands, height, width)
    print(height_array.shape)  # (5, H, W) for a 5-band image

height_array_final = height_array - height_array[height_array > 0].min()
height_array_final = np.clip(height_array_final, 0, height_array_final.max())

## Total volume
height_array_final[0].sum() * 0.018 * 0.018

# Calculation results
# for visualization
# height_array_clipped = (height_array_clipped - height_array_clipped.min()) / (height_array_clipped.max() - height_array_clipped.min())
# height_array_clipped = (height_array_clipped * 255).astype(np.uint8)
# Image.fromarray(height_array_clipped[0])

volume_df = form_volume_df(height_array_final[0])