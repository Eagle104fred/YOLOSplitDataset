# YOLOSplitDataset
# 介绍
本项目只用于分割labelimg生成的yolo格式的数据集, 默认进程池为逻辑核数量, 每个进程处理的任务个数为(单个数据集数量/逻辑核数量)
目录结构:
- root
  - A Dataset
    - images
    - labels
  - B Dataset
    - images
    - labels
  - C Dataset
    - images
    - labels
# Getstarted
开始前请按照目录结构建立好文件夹和规范命名, 本程序放在root目录下运行, 生成的混合数据生成于   root/outFile   。
- 分割比率为 train：val => 9:1，需要修改的可以修改splitRatio这个参数。
- 如果不需要生成混合的train和val数据集请把（def CreateFilePath(self,doNotMixFlag=False): 下的doNotMixFlag设置为True，那样的话会分别为每个数据集生产image和label）
##### Warning: 程序运行前默认清除root/outFile, 请注意酌情备份!!!!!!
##### Warning: 每个数据集下的文件名禁止同名, 否则会相互覆盖!!!!!!
```python
import os
import sys
import random
import shutil
from multiprocessing import Pool,cpu_count
import time

class RootFile:
    def __init__(self,path="./"):
        self.path=path

    def SplitSubRoot(self):
        def GetSubRootList(path):
            originDirList = os.listdir(path)

            originDirList.sort()
            filetedDirList = []
            for dir in originDirList:
                if dir == "outFile":
                    shutil.rmtree(os.path.join(path,dir))
                if os.path.isdir(os.path.join(path, dir)) == True and dir[0]!=".":
                    '''筛选出文件夹'''
                    filetedDirList.append(dir)
            return filetedDirList


        subRootList=GetSubRootList(self.path)

        for subRoot in subRootList:
            print("=========================================")
            print(subRoot + " Spliting")
            print("=========================================")

            subSplit = CreateValFile(subRoot)
            subSplit.Split()



class CreateValFile:
    def __init__(self,path="./",saveFile="outFile",splitRatio=0.1):
        self.path = path
        #self.savePath=self.path+"/"+saveFile
        self.savePath=os.path.join("./",saveFile)
        self.splitRatio = splitRatio


    def CreateFilePath(self,doNotMixFlag=False):

        outValImagePath=""
        outValLabelPath=""
        outTrainImagePath=""
        outTrainLabelPath=""
        imagePath = self.path + "/images"
        labelPath = self.path + "/labels"
        if doNotMixFlag:
            '''判断是否需要分割label和image(默认混合)'''
            outValImagePath = self.savePath + "/val/image"
            outValLabelPath = self.savePath + "/val/label"
            if not os.path.exists(outValImagePath):
                os.makedirs(outValImagePath)
            if not os.path.exists(outValLabelPath):
                os.makedirs(outValLabelPath)

            outTrainImagePath = self.savePath + "/train/image"
            outTrainLabelPath = self.savePath + "/train/label"
            if not os.path.exists(outTrainImagePath):
                os.makedirs(outTrainImagePath)
            if not os.path.exists(outTrainLabelPath):
                os.makedirs(outTrainLabelPath)
        else:
            outValPath = self.savePath + "/val"
            outTrainPath = self.savePath + "/train"
            outValImagePath = outValPath
            outValLabelPath = outValPath
            outTrainImagePath=outTrainPath
            outTrainLabelPath=outTrainPath
            if not os.path.exists(outValPath):
                os.makedirs(outValPath)
            if not os.path.exists(outTrainPath):
                os.makedirs(outTrainPath)
        return imagePath,labelPath,outValImagePath,outValLabelPath,outTrainImagePath,outTrainLabelPath


    def Split(self,multiProcessFlag=True,num_workers=cpu_count(),batch=100):
        '''生成随机抽取数'''
        def CreateSplitList(labelList,splitRatio):
            labelListLen= len(labelList)
            generateCount=labelListLen*splitRatio
            resultNoList=random.sample(range(0,labelListLen-1),int(generateCount))
            resultNoList.sort()
            result=[]
            for i in range(labelListLen):
                if i in resultNoList:
                    result.append(labelList[i])
            #print("Count:",len(result))
            return result

        '''生成输出文件夹'''
        ignore_file = 'classes.txt'
        imagePath,labelPath,outValImagePath,outValLabelPath,outTrainImagePath,outTrainLabelPath\
            =self.CreateFilePath()

        '''过滤/label中的classes.txt文件'''
        originImageList = os.listdir(imagePath)
        originImageList.sort(key=lambda x: (x[:-4]))  # 对文件名按照数字从小到大排序屏蔽最后四位
        originLabelList = os.listdir(labelPath)
        originLabelList.sort(key=lambda x: (x[:-4]))  # 对文件名按照数字从小到大排序屏蔽最后四位
        filtedLabelList = []
        for file in originLabelList:
            if os.path.isfile(os.path.join(labelPath, file)) == True:
                if file != ignore_file and file != sys.argv[0] and file[0]!='.' and file != "outFile":
                    filtedLabelList.append(file)
        fileCount=len(filtedLabelList)
        ''''''
        valList=CreateSplitList(filtedLabelList,0.1)



        if multiProcessFlag:
            print("=========================================")
            print(" MultiProcess ")
            print("=========================================")
            '''多进程池'''
            p=Pool(num_workers)

            '''可选优化: 自动batch(提速效果不太明显)'''
            batch=int(fileCount/num_workers)

            start=0
            end =batch
            sub=SubProcess(originImageList      \
                           ,valList             \
                           ,imagePath           \
                           ,labelPath           \
                           ,outValImagePath     \
                           ,outValLabelPath     \
                           ,outTrainImagePath   \
                           ,outTrainLabelPath)
            stopFlag=False
            while end<=len(filtedLabelList):

                '''分割list'''
                batchList=filtedLabelList[start:end]
                '''滑动窗口'''
                start+=batch
                end+=batch
                if end>=len(filtedLabelList) and stopFlag != True:
                    end=len(filtedLabelList)+1
                    '''保证这个if只会在batch不足的时候运行一次然后跳出循环'''
                    stopFlag=True

                '''为每个子list启动多进程'''
                p.apply_async(sub.Run, args = (batchList,))
                #Run(batchList)

            p.close()
            p.join()

class SubProcess:
    def __init__(self,originImageList,valList,imagePath,labelPath,outValImagePath,outValLabelPath,outTrainImagePath,outTrainLabelPath):
        self.originImageList=originImageList

        self.valList=valList
        self.imagePath=imagePath
        self.labelPath=labelPath
        self.outValLabelPath = outValLabelPath
        self.outValImagePath = outValImagePath
        self.outTrainImagePath = outTrainImagePath
        self.outTrainLabelPath = outTrainLabelPath
    def Run(self,batchList):
        print(" Pid = " + str(os.getpid()))
        for i in range(len(batchList)):
            '''判断/image/中是否有这个文件'''
            valFile = batchList[i]
            for image in self.originImageList:
                imageName = os.path.splitext(image)[0]
                labelName = os.path.splitext(valFile)[0]
                label = labelName + ".txt"
                '''label和image都存在的情况'''
                if (labelName == imageName):
                    '''如果是val则存放在val/否则存放在train/'''
                    if label in self.valList:
                        print(labelName)
                        shutil.copyfile(self.imagePath + "/" + image, self.outValImagePath + "/" + image)
                        shutil.copyfile(self.labelPath + "/" + valFile, self.outValLabelPath + "/" + valFile)
                    else:
                        shutil.copyfile(self.imagePath + "/" + image, self.outTrainImagePath + "/" + image)
                        shutil.copyfile(self.labelPath + "/" + valFile, self.outTrainLabelPath + "/" + valFile)


if __name__ =="__main__":
    t1=time.time()
    print("start-------------------------------------")
    # splitRatio = sys.argv[1]
    #mode = sys.argv[1]  # 第二个参数输入修改的文件目录

    root = RootFile()
    root.SplitSubRoot()


    print("end--------------------------------------")
    t2 = time.time()
    print("Used time: ",t2-t1)
```
