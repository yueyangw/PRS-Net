# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = plt.axes(projection='3d')
with open('data/testfile.pts') as f:
    contents = f.readlines()

# with open('data/04379243/expert_verified/points_label/240ddf8b63318ef534506cc3910614fe.seg') as ff:
#     cate = ff.readlines()

x = np.array([])
y = np.array([])
z = np.array([])
c = np.array([])

for i in range(len(contents)):
    s = contents[i].strip("\n")  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    xx, yy, zz = s.split(' ')
    # cc = cate[i].strip("\n")

    x = np.append(x, float(xx))
    y = np.append(y, float(yy))
    z = np.append(z, float(zz))
    # c = np.append(c, float(cc))

# print(x)

ax.scatter3D(z, x, y)
ax.set_title('3d Scatter plot')
plt.show()