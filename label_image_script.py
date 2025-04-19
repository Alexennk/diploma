import matplotlib.pyplot as plt
import csv

# Load image
img = plt.imread('rgb_image_split.jpg')

# Data structures
coords = []
labels = []

# Temporary buffer for unlabeled points
unlabeled_buffer = []

class_counts = {1: 0, 2: 0}

# Create figure
fig, ax = plt.subplots()
im = ax.imshow(img)
title = ax.set_title('Click up to 200 points, then press "1" or "2" to label them. ESC to finish.')

def update_title():
    title.set_text(
        f'Buffer: {len(unlabeled_buffer)} pts | '
        f'Vegetation (1): {class_counts[1]} | Non-vegetation (2): {class_counts[2]}'
        '\nClick up to 200 points, then press "1" or "2" to label them. ESC to finish.'
    )

def on_click(event):
    if event.button == 1 and event.xdata and event.ydata:
        if len(unlabeled_buffer) < 200:
            x, y = int(event.xdata), int(event.ydata)
            ax.plot(x, y, 'ro', markersize=3)
            unlabeled_buffer.append((x, y))
            plt.draw()
            update_title()
        else:
            print("Buffer full. Press '1' or '2' to label these 200 points.")

def on_key(event):
    if event.key in ['1', '2'] and unlabeled_buffer:
        label = int(event.key)
        coords.extend(unlabeled_buffer)
        labels.extend([label] * len(unlabeled_buffer))
        class_counts[label] += len(unlabeled_buffer)
        print(f"Labeled {len(unlabeled_buffer)} points as class {label}")
        unlabeled_buffer.clear()
        update_title()
    elif event.key == 'escape':
        plt.close()

# Connect events
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

update_title()
plt.show()

# Save labeled points
with open('labeled_points.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'y', 'label'])
    for (x, y), label in zip(coords, labels):
        writer.writerow([x, y, label])

print(f"Saved {len(coords)} labeled points to 'labeled_points.csv'.")
