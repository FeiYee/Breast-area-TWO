import matplotlib.pyplot as plt

labels= ['Benign','Malignant','Normal']

sizes= [438, 210,133]

# explode= (0,0,0,0.1,0,0)

plt.pie(sizes, labels=labels,autopct='%1.1f%%',shadow=True,startangle=150)

plt.title("Percentage of Different Category Images")

plt.show()