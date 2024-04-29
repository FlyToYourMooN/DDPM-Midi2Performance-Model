import matplotlib.pyplot as plt
import numpy as np

# Just to see if the Mel spec and pianoroll correspond

paths = "/disk2/MusicNet/String_Quartet/train/42.npz"
files = np.load(paths)
m, p = files["m"], files["p"]


lens = len(m[0])
ax1 = plt.subplot(2, 1, 1)
plt.imshow(m)

ax2 = plt.subplot(2, 1, 2)
plt.imshow(p)
plt.savefig("mel_piano_align.png", dpi=300)
