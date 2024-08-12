import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("generation_costs.csv")

symbols = ["sample","greedy_search","beam_sample","beam_search"]

for symbol in symbols:
    num_beams = 16
    d = df[df["Type"]==symbol][df["num_beams"]==num_beams]
    x = d["gen_length"]
    y = d["cost"]
    plt.plot(x,y,label=symbol)
    title = f"num_beams={num_beams} cost vs gen_length"
# for symbol in symbols:
#     gen_length = 40
#     d = df[df["Type"]==symbol][df["gen_length"]==gen_length]
#     x = d["num_beams"]
#     y = d["cost"]
#     plt.plot(x,y,label=symbol)
#     title = f"gen_length={gen_length} cost vs num_beams"
    
plt.title(title)
plt.legend()
plt.savefig(f"{title}.png")