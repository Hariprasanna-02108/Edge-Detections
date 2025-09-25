
from tkinter import *
import os
from tkinter import filedialog
import cv2


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


def file_sucess():
    global file_success_screen
    file_success_screen = Toplevel(training_screen)
    file_success_screen.title("File Upload Success")
    file_success_screen.geometry("150x100")
    file_success_screen.configure(bg='pink')
    Label(file_success_screen, text="File Upload Success").pack()
    Button(file_success_screen, text='''ok''', font=(
        'Verdana', 15), height="2", width="30").pack()


global ttype


def training():
    global training_screen

    global clicked

    training_screen = Toplevel(main_screen)
    training_screen.title("Training")
    # login_screen.geometry("400x300")
    training_screen.geometry("600x450+650+150")
    training_screen.minsize(120, 1)
    training_screen.maxsize(1604, 881)
    training_screen.resizable(1, 1)
    training_screen.configure()
    # login_screen.title("New Toplevel")

    Label(training_screen, text='''Upload Image ''', background="#d9d9d9", disabledforeground="#a3a3a3",
          foreground="#000000", width="300", height="2", font=("Calibri", 16)).pack()
    Label(training_screen, text="").pack()


    Button(training_screen, text='''Upload Image''', font=(
        'Verdana', 15), height="2", width="30", command=imgtraining).pack()


def training1():
    global training_screen

    global clicked

    training_screen = Toplevel(main_screen)
    training_screen.title("Training")
    # login_screen.geometry("400x300")
    training_screen.geometry("600x450+650+150")
    training_screen.minsize(120, 1)
    training_screen.maxsize(1604, 881)
    training_screen.resizable(1, 1)
    training_screen.configure()
    # login_screen.title("New Toplevel")

    Label(training_screen, text='''Upload Image ''', background="#d9d9d9", disabledforeground="#a3a3a3",
          foreground="#000000", width="300", height="2", font=("Calibri", 16)).pack()
    Label(training_screen, text="").pack()


    Button(training_screen, text='''Upload Image''', font=(
        'Verdana', 15), height="2", width="30", command=imgtraining1).pack()



def imgtraining():


    import_file_path = filedialog.askopenfilename()

    #image = cv2.imread(import_file_path)
    alpha = 2
    beta = 2
    rho = 0.3
    beginTime = time.time()

    imageGray = preProcessImage(import_file_path)
    pheromone, antRoute, startingPoint = initialize(100, imageGray)
    point, pheromone, antRoute = severalIteration(1000, 100, startingPoint, pheromone, antRoute, imageGray, alpha, beta,
                                                  rho)
    draw(antRoute)
    endTime = time.time()
    runningTime = endTime - beginTime
    print("running time:", runningTime)



def imgtraining1():
    import_file_path = filedialog.askopenfilename()

    # image = cv2.imread(import_file_path)
    image = cv2.imread(import_file_path, cv2.IMREAD_GRAYSCALE)
    import random
    # Fitness function to evaluate the quality of the edges detected
    def fitness_function(edges):
        # The fitness function can be defined as the number of non-zero pixels (edges)
        # or some other metric like SSIM (Structural Similarity Index).
        return np.sum(edges)  # Simple fitness based on the number of edge pixels

    # Apply Canny edge detection
    def apply_edge_detection(threshold1, threshold2):
        return cv2.Canny(image, threshold1, threshold2)

    # Bee Colony Optimization (BCO) algorithm for optimizing Canny thresholds
    def bee_colony_optimization(image, population_size=50, iterations=100, explore_factor=0.3):
        # Initialize the bee population with random parameters (threshold1, threshold2)
        bees = []
        for _ in range(population_size):
            threshold1 = random.randint(50, 150)
            threshold2 = random.randint(150, 300)
            bees.append({'threshold1': threshold1, 'threshold2': threshold2, 'fitness': 0})

        best_solution = None
        best_fitness = -np.inf

        # Perform iterations
        for iteration in range(iterations):
            for bee in bees:
                # Apply edge detection with the current bee's thresholds
                edges = apply_edge_detection(bee['threshold1'], bee['threshold2'])
                # Evaluate fitness
                bee['fitness'] = fitness_function(edges)

                # Update the best solution found
                if bee['fitness'] > best_fitness:
                    best_fitness = bee['fitness']
                    best_solution = bee

            # Exploitation step: bees around the best solution try to improve
            for bee in bees:
                if bee != best_solution:
                    # Slightly adjust the threshold values around the best solution
                    bee['threshold1'] = int(best_solution['threshold1'] + random.randint(-5, 5))
                    bee['threshold2'] = int(best_solution['threshold2'] + random.randint(-5, 5))

            # Exploration step: some bees explore new areas
            for bee in bees:
                if random.random() < explore_factor:
                    bee['threshold1'] = random.randint(50, 150)
                    bee['threshold2'] = random.randint(150, 300)

            print(f"Iteration {iteration + 1}/{iterations}: Best Fitness = {best_fitness}")

        return best_solution

    # Running BCO to optimize edge detection
    best_solution = bee_colony_optimization(image)

    # Apply edge detection with the optimized thresholds
    final_edges = apply_edge_detection(best_solution['threshold1'], best_solution['threshold2'])

    # Show the result
    plt.subplot(1, 2, 1)
    plt.title("Original Image")
    plt.imshow(image, cmap='gray')

    plt.subplot(1, 2, 2)
    plt.title("Optimized Edge Detection")
    plt.imshow(final_edges, cmap='gray')

    plt.show()

    # Print the best thresholds found
    print(f"Best Thresholds: threshold1 = {best_solution['threshold1']}, threshold2 = {best_solution['threshold2']}")


def main_account_screen():
    global main_screen
    main_screen = Tk()
    width = 600
    height = 600
    screen_width = main_screen.winfo_screenwidth()
    screen_height = main_screen.winfo_screenheight()
    x = (screen_width / 2) - (width / 2)
    y = (screen_height / 2) - (height / 2)
    main_screen.geometry("%dx%d+%d+%d" % (width, height, x, y))
    main_screen.resizable(0, 0)
    # main_screen.geometry("300x250")
    main_screen.configure()
    main_screen.title(" Image Edge Detection")

    Label(text="Image Edge Detection", width="300", height="5", font=("Calibri", 16)).pack()

    Button(text="Ant-Colony EdgeDetection", font=(
        'Verdana', 15), height="2", width="30", command=training, highlightcolor="black").pack(side=TOP)
    Label(text="").pack()
    Button(text="Bee-Colony EdgeDetection", font=(
        'Verdana', 15), height="2", width="30", command=training1, highlightcolor="black").pack(side=TOP)

    Label(text="").pack()

    main_screen.mainloop()


main_account_screen()
