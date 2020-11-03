import csv

test = open("./ImageSets/Segmentation/test.txt", mode = "w")
train = open("./ImageSets/Segmentation/train.txt", mode = "w")
trainval = open("./ImageSets/Segmentation/trainval.txt", mode = "w")
val = open("./ImageSets/Segmentation/val.txt", mode = "w")

def path2number(path):
    mark = path.index('/')
    mark = path.index('/', mark+1)
    return path[mark + 1: -4]


with open("./train_standard_annotation.csv",mode='r',newline='') as f:
    reader = csv.reader(f)
    result = list(reader)
    last = "0"
    for i in range(0, 2098):
        number = path2number(result[i][0])
        if number == last:
            continue
        last = number
        if (int(number) >= 1 and int(number) <= 50) \
           or (int(number) >= 151 and int(number) <= 200) \
           or (int(number) >= 501 and int(number) <= 550):
            test.write(number)
            test.write("\n")
        else:
            trainval.write(number)
            trainval.write("\n")
            if int(number) % 10 == 1 or int(number) % 10 == 2:
                val.write(number)
                val.write("\n")
            else:
                train.write(number)
                train.write("\n")


