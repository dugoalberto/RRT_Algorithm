import matplotlib
matplotlib.use('TkAgg')

from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt



# Apri l'immagine
img = Image.open('cspace.png')

# Converti l'immagine in scala di grigi
img = ImageOps.grayscale(img)

# Converti l'immagine in un array NumPy
np_img = np.array(img)

# Inverti il bianco e il nero
np_img = ~np_img
np_img[np_img > 0] = 1

# Imposta la mappa dei colori su "binary" (bianco e nero)
plt.set_cmap('binary')

# Visualizza l'immagine
plt.imshow(np_img)

# Salva l'immagine
np.save('cspace.npy', np_img)

# Carica l'immagine salvata
grid = np.load('cspace.npy')

# Imposta nuovamente la mappa dei colori su "binary"
plt.set_cmap('binary')

# Visualizza l'immagine caricata
plt.imshow(grid)

# Aggiusta il layout
plt.tight_layout()

# Mostra l'immagine
plt.show()

