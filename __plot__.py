import pandas as pd
import os
import matplotlib.pyplot as plt

dir = os.listdir("logs")
dir.sort()
latestfile = "logs/" + dir[-1]

data = pd.read_csv(latestfile,
                   names=["Time",
                          "Loss"]
                   )

print("Total Time (sec): ", data["Time"].iloc[-1])
print("Final Loss: ", data["Loss"].iloc[-1])

plt.plot(data["Time"], data["Loss"])
plt.show()

