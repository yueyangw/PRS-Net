# from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax = plt.axes(projection='3d')
plt.xlim((0, 32))
plt.ylim((0, 32))
ax.set_zlim((0, 32))
with open('../data/ShapeNet/02773838/8bc53bae41a8105f5c7506815f553527/models/model_normalized.obj') as f:
    contents = f.readlines()

# with open('data/04379243/expert_verified/points_label/240ddf8b63318ef534506cc3910614fe.seg') as ff:
#     cate = ff.readlines()

x = np.array([])
y = np.array([])
z = np.array([])
c = np.array([])

for i in range(len(contents)):
    s = contents[i].strip("\n")  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
    try:
        c, xx, yy, zz = s.split(' ')
    except:
        continue
    if c != 'v':
        continue
    # cc = cate[i].strip("\n")

    x = np.append(x, (float(xx) + 0.4) * 5 / 4 * 32)
    y = np.append(y, (float(yy) + 0.4) * 5 / 4 * 32)
    z = np.append(z, (float(zz) + 0.4) * 5 / 4 * 32)
    # c = np.append(c, float(cc))

# print(x)

ax.scatter3D(x, y, z)
ax.set_title('3d Scatter plot')
plt.show()
