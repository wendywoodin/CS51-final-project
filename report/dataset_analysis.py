import numpy as np
from loader import Loader
import matplotlib.pyplot as plt

load = Loader("testing")

# Load in the digits from the dataset, each in their own array
zeros, labels = load.load_dataset([0])
ones, labels = load.load_dataset([1])
twos, labels = load.load_dataset([2])
threes, labels = load.load_dataset([3])
fours, labels = load.load_dataset([4])
fives, labels = load.load_dataset([5])
sixes, labels = load.load_dataset([6])
sevens, labels = load.load_dataset([7])
eights, labels = load.load_dataset([8])
nines, labels = load.load_dataset([9])


# Find the minimum number of digits so we don't go outside
# the bounds of the array
lengths = [len(zeros),len(ones),len(twos),len(threes),\
    len(fours),len(fives),len(sixes),len(sevens),len(eights)]

length = min(lengths)


# Set up checking for similarity level between the same digits
sim0 = []
sim1 = []
sim2 = []
sim3 = []
sim4 = []
sim5 = []
sim6 = []
sim7 = []
sim8 = []
sim9 = []

for l in range(length-1):
    sim0.append(np.linalg.norm(zeros[l]-zeros[l+1]))
    sim1.append(np.linalg.norm(ones[l]-ones[l+1]))
    sim2.append(np.linalg.norm(twos[l]-twos[l+1]))
    sim3.append(np.linalg.norm(threes[l]-threes[l+1]))
    sim4.append(np.linalg.norm(fours[l]-fours[l+1]))
    sim5.append(np.linalg.norm(fives[l]-fives[l+1]))
    sim6.append(np.linalg.norm(sixes[l]-sixes[l+1]))
    sim7.append(np.linalg.norm(sevens[l]-sevens[l+1]))
    sim8.append(np.linalg.norm(eights[l]-eights[l+1]))
    sim9.append(np.linalg.norm(nines[l]-nines[l+1]))

# Getting the average of the differences for each digit
similarities_same_digits = [np.mean(sim0), np.mean(sim1), \
    np.mean(sim2), np.mean(sim3), np.mean(sim4), np.mean(sim5), \
    np.mean(sim6), np.mean(sim7), np.mean(sim8), np.mean(sim9)]

# Getting overall average, min and max of the differences
av_sim_same = np.mean(similarities_same_digits)
# 126.19218758152313

min_sim_same = min(similarities_same_digits)
# 91.708073557284379

max_sim_same = max(similarities_same_digits)
# 137.15349426741975

# For graphing purposes
x_axis_same = [1.,1.,1.,1.,1.,1.,1.,1.,1.,1.]


# Set up checking for similarity between similar but not same digits
diff_2_3 = []
diff_4_9 = []
diff_7_9 = []
diff_5_6 = []
diff_8_9 = []
diff_0_9 = []
diff_2_7 = []
diff_8_6 = []

for l in range(length-1):
    diff_2_3.append(np.linalg.norm(twos[l]-threes[l+1]))
    diff_4_9.append(np.linalg.norm(fours[l]-fives[l+1]))
    diff_7_9.append(np.linalg.norm(sevens[l]-nines[l+1]))
    diff_5_6.append(np.linalg.norm(fives[l]-sixes[l+1]))
    diff_8_9.append(np.linalg.norm(eights[l]-nines[l+1]))
    diff_0_9.append(np.linalg.norm(zeros[l]-nines[l+1]))
    diff_2_7.append(np.linalg.norm(twos[l]-sevens[l+1]))
    diff_8_6.append(np.linalg.norm(eights[l]-sixes[l+1]))

# Getting the average of the differences for each pair of digits
diffs_sim_digits = [np.mean(diff_2_3), np.mean(diff_4_9), \
    np.mean(diff_7_9), np.mean(diff_5_6), np.mean(diff_8_9), \
    np.mean(diff_0_9), np.mean(diff_2_7), np.mean(diff_8_6)]

# Getting overall average, min and max of the differences
av_diff_sim = np.mean(diffs_sim_digits)
# 134.02751258579383

min_diff_sim = min(diffs_sim_digits)
# 124.84746353394799

max_diff_sim = max(diffs_sim_digits)
# 138.44045771113068

# For graphing purposes
x_axis_diff = [1.,1.,1.,1.,1.,1.,1.,1.]

plt.plot(x_axis_same, similarities_same_digits, 'bo', x_axis_diff, diffs_sim_digits, 'rs')
plt.xlabel('same digits = blue, different digits = red')
plt.ylabel('average distance')
plt.title('Average distances between different digits')
plt.show()

