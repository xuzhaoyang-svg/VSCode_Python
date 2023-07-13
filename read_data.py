from torch.utils.data import Dataset
from PIL import Image
import os

class MyData(Dataset):

    # root_dir为输入的数据，label_dir为数据标签
    def __init__(self, root_dir, label_dir):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(root_dir, label_dir) #数据路径
        self.img_path = os.listdir(self.path) #读取到的数据列表

    # 获取每个数据及其label
    def __getitem__(self, idx):
        img_name = self.img_path[idx]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name) #单个数据的路径
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


# 测试
root_dir = '/Users/marie/Downloads/数据集/hymenoptera_data/train'
ants_label_dir =  'ants'
bees_label_dir =  'bees'

ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))

img, label = train_dataset[124]

img.show()
