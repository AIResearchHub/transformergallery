import pandas as pd
import os
import matplotlib.pyplot as plt

dir = os.listdir("logs")
dir.sort()
latestfile = "logs/" + dir[-1]

data = pd.read_csv(latestfile,
                   names=["Loss"]
                   )

plt.subplot(1, 1, 1)
plt.plot(data["Loss"])

plt.show()

