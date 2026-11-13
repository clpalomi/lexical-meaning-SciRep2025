"""
@author: u164233
"""

# IMPORT PACKAGES
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image


# PICTURES
# German:  pict 1: boy, pict 2: countryside, pict 4: couple, pict 6: old woman
# Spanish: pict 1: countryside, pict2: couple, pict3: bridge


# PATH INPUT
path_input     = '/PATH/TO/PRIVATE_DATA/german/texts/'
path_demo      = '/PATH/TO/PRIVATE_DATA/german/'
path_results   = '/PATH/TO/RESULTS/'
path_santander = '/PATH/TO/PRIVATE_DATA/spanish/'
path_london    = '/PATH/TO/PRIVATE_DATA/english/'


""" COMBINE FIGURES IN PANEL """
matplotlib.rcParams['font.family'] = 'Times New Roman'

# FIGURE 5

# Load images
img1 = Image.open(path_results+'heatmap_corr_German_SSD.jpg')
img2 = Image.open(path_results+'heatmap_corr_German_HC.jpg')
img3 = Image.open(path_results+'heatmap_corr_Spanish_SSD.jpg')
img4 = Image.open(path_results+'heatmap_corr_Spanish_HC.jpg')
img5 = Image.open(path_results+'heatmap_corr_English_FEP.jpg')
img6 = Image.open(path_results+'heatmap_corr_English_HC.jpg')

# Create 3 rows, 2 columns panel
fig, axes = plt.subplots(3, 2, figsize=(8, 12), gridspec_kw={'hspace': -0.4, 'wspace': 0.01})

# Flatten the axes array and define the labels
axes = axes.flatten()
labels = ['A', 'B', 'C', 'D', 'E', 'F']
images = [img1, img2, img3, img4, img5, img6]

# Display each image in the corresponding subplot
for ax, img, label in zip(axes, images, labels):
    ax.imshow(img)
    ax.axis('off')
    ax.text(0.01, 0.99, label, transform=ax.transAxes,
            fontsize=15, fontweight='bold', va='top', ha='left')

# Save and show figure
plt.savefig(path_results+"Figure5.jpg", dpi=1000, bbox_inches='tight', pad_inches=0.1)
plt.show()
plt.close()


