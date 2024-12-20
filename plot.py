import matplotlib.pyplot as plt

# Loss values for each epoch for hangman_mlm_high.pt
loss_high = [
    2.6634, 2.4635, 2.4035, 2.3703, 2.3483, 2.3283, 2.3142, 2.3019, 2.2919, 2.2823,
    2.2754, 2.2678, 2.2613, 2.2542, 2.2475, 2.2445, 2.2378, 2.2349, 2.2305, 2.2251,
    2.2227, 2.2196, 2.2156, 2.2122, 2.2098, 2.2064, 2.2048, 2.1990, 2.2012, 2.1968,
    2.1931, 2.1903, 2.1893, 2.1864, 2.1827, 2.1809, 2.1776, 2.1780, 2.1771, 2.1745,
    2.1706, 2.1699, 2.1685, 2.1669, 2.1659, 2.1626, 2.1621, 2.1590, 2.1590, 2.1568,
    2.1533, 2.1548, 2.1543, 2.1537, 2.1522, 2.1500, 2.1471, 2.1464, 2.1451, 2.1452,
    2.1434, 2.1415, 2.1395, 2.1420, 2.1396, 2.1387, 2.1382, 2.1352, 2.1355, 2.1342,
    2.1323, 2.1325, 2.1323, 2.1301, 2.1295, 2.1277, 2.1286, 2.1275, 2.1268, 2.1242,
    2.1234, 2.1249, 2.1245, 2.1253, 2.1203, 2.1196, 2.1192, 2.1178, 2.1197, 2.1176,
    2.1175, 2.1176, 2.1175, 2.1148, 2.1139, 2.1129, 2.1110, 2.1115, 2.1116, 2.1110
]

# Loss values for each epoch for hangman_mlm_medium.pt
loss_medium = [
    2.5875, 2.3045, 2.2230, 2.1761, 2.1435, 2.1170, 2.0964, 2.0783, 2.0624, 2.0494,
    2.0384, 2.0246, 2.0180, 2.0075, 1.9952, 1.9893, 1.9826, 1.9768, 1.9731, 1.9638,
    1.9604, 1.9544, 1.9478, 1.9445, 1.9394, 1.9326, 1.9289, 1.9266, 1.9207, 1.9161,
    1.9152, 1.9097, 1.9055, 1.9039, 1.8990, 1.8976, 1.8930, 1.8900, 1.8869, 1.8846,
    1.8810, 1.8793, 1.8759, 1.8722, 1.8716, 1.8652, 1.8665, 1.8619, 1.8610, 1.8610,
    1.8561, 1.8557, 1.8545, 1.8545, 1.8483, 1.8465, 1.8417, 1.8430, 1.8416, 1.8386,
    1.8390, 1.8372, 1.8317, 1.8333, 1.8315, 1.8288, 1.8300, 1.8280, 1.8254, 1.8242,
    1.8217, 1.8233, 1.8194, 1.8170, 1.8129, 1.8152, 1.8101, 1.8113, 1.8118, 1.8080,
    1.8081, 1.8083, 1.8040, 1.8069, 1.8031, 1.8033, 1.8009, 1.8011, 1.7978, 1.7989,
    1.7982, 1.7931, 1.7936, 1.7927, 1.7931, 1.7922, 1.7903, 1.7889, 1.7846, 1.7878
]

# Loss values for each epoch for hangman_mlm_low.pt
loss_low = [
    2.5699, 2.2011, 2.0873, 2.0209, 1.9804, 1.9431, 1.9128, 1.8927, 1.8735, 1.8508,
    1.8357, 1.8232, 1.8100, 1.7960, 1.7818, 1.7718, 1.7589, 1.7520, 1.7433, 1.7384,
    1.7259, 1.7234, 1.7113, 1.7072, 1.6965, 1.6925, 1.6875, 1.6812, 1.6776, 1.6667,
    1.6627, 1.6641, 1.6563, 1.6536, 1.6523, 1.6427, 1.6357, 1.6328, 1.6334, 1.6281,
    1.6214, 1.6211, 1.6165, 1.6135, 1.6044, 1.6085, 1.6041, 1.5980, 1.5942, 1.5928,
    1.5900, 1.5904, 1.5793, 1.5786, 1.5781, 1.5747, 1.5710, 1.5671, 1.5645, 1.5669,
    1.5638, 1.5591, 1.5581, 1.5555, 1.5479, 1.5532, 1.5462, 1.5487, 1.5433, 1.5447,
    1.5440, 1.5377, 1.5344, 1.5307, 1.5315, 1.5340, 1.5269, 1.5285, 1.5249, 1.5186,
    1.5189, 1.5203, 1.5168, 1.5183, 1.5125, 1.5137, 1.5079, 1.5102, 1.5084, 1.5071,
    1.5066, 1.5001, 1.4968, 1.5011, 1.4985, 1.4983, 1.4945, 1.4964, 1.4970, 1.4947
]

# Loss values for each epoch for hangman_mlm_short.pt
loss_short = [
    2.8486, 2.6266, 2.5367, 2.4869, 2.4500, 2.4282, 2.4143, 2.3934, 2.3897, 2.3806,
    2.3740, 2.3625, 2.3533, 2.3485, 2.3395, 2.3321, 2.3340, 2.3235, 2.3225, 2.3162,
    2.3147, 2.3095, 2.3034, 2.2974, 2.3008, 2.2942, 2.2911, 2.2884, 2.2901, 2.2871,
    2.2806, 2.2811, 2.2718, 2.2766, 2.2781, 2.2723, 2.2661, 2.2640, 2.2668, 2.2607,
    2.2654, 2.2645, 2.2568, 2.2509, 2.2520, 2.2564, 2.2542, 2.2504, 2.2488, 2.2465,
    2.2504, 2.2412, 2.2410, 2.2483, 2.2352, 2.2444, 2.2322, 2.2390, 2.2364, 2.2294,
    2.2289, 2.2323, 2.2271, 2.2289, 2.2356, 2.2259, 2.2274, 2.2223, 2.2247, 2.2198,
    2.2251, 2.2203, 2.2164, 2.2170, 2.2140, 2.2162, 2.2077, 2.2146, 2.2091, 2.2103,
    2.2179, 2.2112, 2.2161, 2.2104, 2.2047, 2.2078, 2.2118, 2.2005, 2.2050, 2.2041,
    2.1988, 2.1979, 2.2016, 2.2044, 2.1990, 2.1996, 2.1952, 2.1952, 2.1971, 2.1901
]

# Epoch numbers
epochs = list(range(1, 101))

# Function to plot loss
def plot_loss(epochs, loss, model_name, color):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Loss', color=color)
    plt.title(f'Training Loss over Epochs for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Function to plot and save loss
def plot_loss_save(epochs, loss, model_name, color, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, loss, label='Loss', color=color)
    plt.title(f'Training Loss over Epochs for {model_name}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Plot and save individual loss curves
plot_loss(epochs, loss_high, 'hangman_mlm_high.pt', 'blue')
plot_loss(epochs, loss_medium, 'hangman_mlm_medium.pt', 'green')
plot_loss(epochs, loss_low, 'hangman_mlm_low.pt', 'red')
plot_loss(epochs, loss_short, 'hangman_mlm_short.pt', 'purple')

plot_loss_save(epochs, loss_high, 'hangman_mlm_high.pt', 'blue', 'loss_high.png')
plot_loss_save(epochs, loss_medium, 'hangman_mlm_medium.pt', 'green', 'loss_medium.png')
plot_loss_save(epochs, loss_low, 'hangman_mlm_low.pt', 'red', 'loss_low.png')
plot_loss_save(epochs, loss_short, 'hangman_mlm_short.pt', 'purple', 'loss_short.png')

# Plot and save combined loss curves
plt.figure(figsize=(14, 8))
plt.plot(epochs, loss_high, label='High Model', color='blue')
plt.plot(epochs, loss_medium, label='Medium Model', color='green')
plt.plot(epochs, loss_low, label='Low Model', color='red')
plt.plot(epochs, loss_short, label='Short Model', color='purple')

plt.title('Training Loss over Epochs for All Models')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_all_models.png')
plt.show()  # Optional: Remove if you don't want to display the plot
plt.close()