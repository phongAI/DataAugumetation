from torchvision import utils,transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torch
import os
import uuid
import time

#Parameter
repeat = 1
augumentedFolderName = "AugumentedFolder"
sourceFolder = "../Dataset/LeafData/train"
input_size = (224,224)
batch_size = 8

def calStdMean(path):
    if os.path.exists(path):
        """Load data"""
        dataset = ImageFolder(root=path,transform=transforms.ToTensor())
        dataLoader = DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)
        channels_sum, channel_squared_sum, num_batches = 0,0,0
        for data,_ in dataLoader:
            channels_sum += torch.mean(data,dim=[0,2,3])
            channel_squared_sum += torch.mean(data**2,dim=[0,2,3])
            num_batches += 1
        mean = channels_sum/num_batches
        std = (channel_squared_sum/num_batches - mean**2)**0.5
        print(f"Mean value is : {mean.numpy()}")
        print(f"Std value is : {std.numpy()}")
        return mean,std
    else:
        raise Exception("Folder is valid!")

def main():
    beginTime = time.time()
    """Create save folder"""
    os.makedirs(augumentedFolderName,exist_ok=True)

    """Calculate mean and standard"""
    mean, std = calStdMean(sourceFolder)

    """Transform"""
    transform = transforms.Compose(transforms=[transforms.ToTensor(),
                                               transforms.RandomRotation(degrees=15),
                                               transforms.RandomErasing(p=0.2),
                                               transforms.Normalize(mean=mean,std=std),
                                               transforms.Resize(size=input_size),
                                               #Add more transform here
                                               ])

    """Get subdirectory name then create it"""
    subFolderName = os.path.basename(os.path.dirname(sourceFolder))
    subFolderPath = os.path.join(augumentedFolderName,subFolderName)
    os.makedirs(subFolderPath,exist_ok=True)

    """Do augument data for repeat times:"""
    imageData = ImageFolder(root=sourceFolder,transform=transform)
    for turn in range(repeat):
        for (data,label) in imageData:
            imageName = os.path.join(subFolderPath,str(uuid.uuid4())+".jpg")
            utils.save_image(data,fp=imageName)
    print("Save done")
    duration = time.time() - beginTime
    print(f"Time for augumenting:{round(duration,1)}s")
if __name__ == "__main__":
    main()