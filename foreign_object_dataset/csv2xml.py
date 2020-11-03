import os
import cv2
import csv

from lxml import etree

class GEN_Annotations:
    def __init__(self, filename):
        self.root = etree.Element("annotation")
        child1 = etree.SubElement(self.root, "folder")
        child1.text = "VOC2007"
        child2 = etree.SubElement(self.root, "filename")
        child2.text = filename
        child3 = etree.SubElement(self.root, "source")
        child31 = etree.SubElement(child3, "database")
        child31.text = "My Database"
        child32 = etree.SubElement(child3, "annotation")
        child32.text = "VOC2007"
        child33 = etree.SubElement(child3, "image")
        child33.text = "flickr"
        child34 = etree.SubElement(child3, "flickrid")
        child34.text = "NULL"
        child4 = etree.SubElement(self.root, "owner")
        child41 = etree.SubElement(child4, "flickrid")
        child41.text = "NULL"
        child42 = etree.SubElement(child4, "name")
        child42.text = "idaneel"


    def set_size(self, height, witdh, channel):
        size = etree.SubElement(self.root, "size")
        widthn = etree.SubElement(size, "width")
        widthn.text = str(witdh)
        heightn = etree.SubElement(size, "height")
        heightn.text = str(height)
        channeln = etree.SubElement(size, "depth")
        channeln.text = str(channel)

    def set_segmented(self, segment):
        sgmt = etree.SubElement(self.root, "segmented")
        sgmt.text = str(segment)

    def add_pic_attr(self, label, xmin, ymin, xmax, ymax):
        objectn = etree.SubElement(self.root, "object")
        namen = etree.SubElement(objectn, "name")
        namen.text = label
        pose = etree.SubElement(objectn, "pose")
        pose.text = str("Unspecified")
        truncated = etree.SubElement(objectn, "truncated")
        truncated.text = str(0)
        difficult = etree.SubElement(objectn, "difficult")
        difficult.text = str(0)
        bndbox = etree.SubElement(objectn, "bndbox")
        xminn = etree.SubElement(bndbox, "xmin")
        xminn.text = str(xmin)
        yminn = etree.SubElement(bndbox, "ymin")
        yminn.text = str(ymin)
        xmaxn = etree.SubElement(bndbox, "xmax")
        xmaxn.text = str(xmax)
        ymaxn = etree.SubElement(bndbox, "ymax")
        ymaxn.text = str(ymax)

    
    def savefile(self,filename):
        tree = etree.ElementTree(self.root)
        tree.write(filename, pretty_print=True, xml_declaration=False, encoding='utf-8')

    
    def genvoc(filename,class_,width,height,depth,xmin,ymin,xmax,ymax,savedir):
        anno = GEN_Annotations(filename)
        anno.set_size(width,height,depth)
        anno.add_pic_attr("pos", xmin, ymin, xmax, ymax)
        anno.savefile(savedir)

def path2number(path):
    mark = path.index('/')
    mark = path.index('/', mark+1)
    return path[mark + 1: -4]

def read_jpg(path):
    img = cv2.imread(path)
    return len(img), len(img[0]), len(img[0][0])

if __name__ == '__main__':
    with open("./train_standard_annotation.csv",mode='r',newline='') as f:
        reader = csv.reader(f)
        result = list(reader)
        filename = "%s.jpg"
        path2save = os.path.join(".","Annotations", "%s.xml")
        path2jpg = os.path.join(".", "JPEGImages", filename)

        last = '1'
        height, width, chanael = read_jpg(path2jpg%last)

        anno = GEN_Annotations(filename%last)
        anno.set_size(height,width,chanael)
        anno.set_segmented(0)
        for i in range(0,len(result)):
            if path2number(result[i][0]) != last:
                # print(path2save%last)
                anno.savefile(path2save%last)
                last = path2number(result[i][0])
                anno = GEN_Annotations(filename%last)
                height, width, chanael = read_jpg(path2jpg%last)
                if height != 1440 or width != 2560:
                    print(last)
                anno.set_size(height,width,chanael)
                anno.set_segmented(0)

            xmin = result[i][1]
            ymin = result[i][2]
            xmax = result[i][3]
            ymax = result[i][4]
            label = result[i][5]
            anno.add_pic_attr(label, xmin, ymin, xmax, ymax)
        anno.savefile(path2save%last)

