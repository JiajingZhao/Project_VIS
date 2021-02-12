import numpy as np
import matplotlib.pyplot as plt

from habitat_sim.utils.data import ImageExtractor
#%set_env DISPLAY=:0

# For viewing the extractor output
def display_sample(sample):
    img = sample["rgba"]
    depth = sample["depth"]
    semantic = sample["semantic"]

    arr = [img, depth, semantic]
    titles = ["rgba", "depth", "semantic"]
    plt.figure(figsize=(12, 8))
    for i, data in enumerate(arr):
        ax = plt.subplot(1, 3, i + 1)
        ax.axis("off")
        ax.set_title(titles[i])
        plt.imshow(data)

    plt.show()

scene_filepath = "/home/habitat/habitat/habitat-sim/data/scene_datasets/mp3d/YVUC4YcDtcY/YVUC4YcDtcY.glb"

extractor = ImageExtractor(
    scene_filepath,
    img_size=(512, 512),
    output=["rgba", "depth", "semantic"],
)

# Use the list of train outputs instead of the default, which is the full list
# of outputs (test + train)
extractor.set_mode('full')

# Index in to the extractor like a normal python list
sample = extractor[0]

# Or use slicing
samples = extractor[1:4]
samples = np.array(samples, dtype="object")
for sample in samples:
    display_sample(sample)

# Close the extractor so we can instantiate another one later
# (see close method for detailed explanation)
extractor.close()