import numpy as np
import cv2 as cv
import math
import matplotlib.pyplot as plt
import time

def preProcessImage(filePath):
    imageOriginal=cv.imread(filePath)
    imageResize=cv.resize(imageOriginal,(200,200))
    imageGray=cv.cvtColor(imageResize,cv.COLOR_BGR2GRAY)
    # cv.imshow("pic",imageGray)
    # cv.waitKey(0)
    return imageGray

def distanceCalculation(pixelX,pixelY,imageGray):
    distance=np.zeros(8)
    pixelX=int(pixelX)
    pixelY=int(pixelY)
    step=1
    distance[0]=math.fabs(int(imageGray[pixelX-step][pixelY-step])-int(imageGray[pixelX][pixelY]))
    distance[1]=math.fabs(int(imageGray[pixelX][pixelY-step])-int(imageGray[pixelX][pixelY]))
    distance[2]=math.fabs(int(imageGray[pixelX+step][pixelY-step])-int(imageGray[pixelX][pixelY]))
    distance[3]=math.fabs(int(imageGray[pixelX-step][pixelY])-int(imageGray[pixelX][pixelY]))
    distance[4]=math.fabs(int(imageGray[pixelX+step][pixelY])-int(imageGray[pixelX][pixelY]))
    distance[5]=math.fabs(int(imageGray[pixelX-step][pixelY+step])-int(imageGray[pixelX][pixelY]))
    distance[6]=math.fabs(int(imageGray[pixelX][pixelY+step])-int(imageGray[pixelX][pixelY]))
    distance[7]=math.fabs(int(imageGray[pixelX+step][pixelY+step])-int(imageGray[pixelX][pixelY]))
    return distance

def initialize(antNum,imageGray):

    count=0
    distance=np.zeros(8)
    startingPoint = np.zeros(2)
    while(count==0):

        startingPoint[0] = np.random.randint(20, 180, 1)
        startingPoint[1] = np.random.randint(20, 180, 1)
        startingPoint = np.array(startingPoint)
        distance=distanceCalculation(startingPoint[0],startingPoint[1],imageGray)
        for i in range(8):
            if distance[i]>20:
                count+=1

    pheromone=np.ones((antNum,8))

    antRoute=np.zeros((antNum,2))
    return pheromone,antRoute,startingPoint


def probabilityCalculation(antIndex,distance,pheromone,alpha,beta):
    probability=np.zeros(8)
    for i in range(8):
        probability[i]=pheromone[antIndex][i]**alpha+distance[i]**beta
    probabilitySum=np.sum(probability)
    probability=probability/probabilitySum
    return probability


def roulette(probability):
    probabilityTotal = np.zeros(len(probability))
    probabilityTmp = 0
    for i in range(len(probability)):
        probabilityTmp += probability[i]
        probabilityTotal[i] = probabilityTmp
    randomNumber=np.random.rand()
    result=0
    for i in range(1, len(probabilityTotal)):
        if randomNumber<probabilityTotal[0]:
            result=0
            break
        elif probabilityTotal[i - 1] < randomNumber <= probabilityTotal[i]:
            result=i
    return result

def singleTransfer(antIndex,startingPoint,pheromone,antRoute,imageGray,alpha,beta,rho):
    antRoute[antIndex][0]=startingPoint[0]
    antRoute[antIndex][1]=startingPoint[1]

    distance=distanceCalculation(startingPoint[0],startingPoint[1],imageGray)

    probability=probabilityCalculation(antIndex,distance,pheromone,alpha,beta)

    nextPixel=roulette(probability)
    nextPoint=np.zeros(2)
    step= 1
    if nextPixel==0:
        nextPoint[0]=startingPoint[0]-step
        nextPoint[1]=startingPoint[1]-step
        if nextPoint[0]<step:
            nextPoint[0]=step
        elif nextPoint[0]>=197-step:
            nextPoint[0]=197-step
        if nextPoint[1]<step:
            nextPoint[1]=step
        elif nextPoint[1]>=197-step:
            nextPoint[1]=197-step
    elif nextPixel==1:
        nextPoint[0]=startingPoint[0]
        nextPoint[1]=startingPoint[1]-step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==2:
        nextPoint[0]=startingPoint[0]+step
        nextPoint[1]=startingPoint[1]-step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] =step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==3:
        nextPoint[0]=startingPoint[0]-step
        nextPoint[1]=startingPoint[1]
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==4:
        nextPoint[0]=startingPoint[0]+step
        nextPoint[1]=startingPoint[1]
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==5:
        nextPoint[0]=startingPoint[0]-step
        nextPoint[1]=startingPoint[1]+step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==6:
        nextPoint[0]=startingPoint[0]
        nextPoint[1]=startingPoint[1]+step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step
    elif nextPixel==7:
        nextPoint[0]=startingPoint[0]+step
        nextPoint[1]=startingPoint[1]+step
        if nextPoint[0] < step:
            nextPoint[0] = step
        elif nextPoint[0] >= 197-step:
            nextPoint[0] = 197-step
        if nextPoint[1] < step:
            nextPoint[1] = step
        elif nextPoint[1] >= 197-step:
            nextPoint[1] = 197-step

    count=0
    for i in range(8):
        if distance[i]<20:
            count+=1
    if count>7 or count<1:
        nextPoint=startingPoint
    else:
        # 更新信息素
        deltaPheromone = np.zeros(8)
        for i in range(8):
            if distance[i] > 20:
                deltaPheromone[i] = distance[i] / 255
        deltaPheromoneSum = np.sum(deltaPheromone)
        for i in range(8):
            pheromone[antIndex][i] = (1 - rho) * pheromone[antIndex][i] + deltaPheromoneSum
    return nextPoint, pheromone, antRoute

def singleIteration(antNum,point,pheromone,antRoute,imageGray,alpha,beta,rho):
    pheromone_,antRoute_,point=initialize(antNum,imageGray)
    for i in range(antNum):
        point,pheromone,antRoute=singleTransfer(i,point,pheromone,antRoute,imageGray,alpha,beta,rho)
    return point,pheromone,antRoute


def severalIteration(iterateTimes,antNum,point,pheromone,antRoute,imageGray,alpha,beta,rho):
    for i in range(iterateTimes):
        print("iterate ",i,":")
        point,pheromone,antRoute=singleIteration(antNum,point,pheromone,antRoute,imageGray,alpha,beta,rho)
        draw(antRoute)
    plt.show()
    return point,pheromone,antRoute


def draw(antRoute):
    plt.plot(antRoute[:, 0], antRoute[:, 1],color="blue")
    imageDetection = np.zeros((200, 200))
    for i in range(200):
        for j in range(200):
            imageDetection[i][j] = 0
    for i in range(len(antRoute)):
        x = int(antRoute[i][0])
        y = int(antRoute[i][1])
        imageDetection[x][y] = 255


alpha=2
beta=2
rho=0.3
beginTime=time.time()

imageGray=preProcessImage("coin.jpg")
pheromone,antRoute,startingPoint=initialize(100,imageGray)
point,pheromone,antRoute=severalIteration(1000,100,startingPoint,pheromone,antRoute,imageGray,alpha,beta,rho)
draw(antRoute)
endTime=time.time()
runningTime=endTime-beginTime
print("running time:",runningTime)
