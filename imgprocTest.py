import imgproc
from PIL import Image

#*******************************
n_testfiles=2
n_rows=4            #each row has 6 samples

#************************ TEST DATA **************

name="testData/"+"test"
ext=".jpg"
for i in range(0,n_testfiles):            
    imagename=name+str(i)+ext
    im=Image.open(imagename)
    
    for j in range(0,n_rows):        #no. of samples(rows,cols)
        y=60+j*150
        for k in range(0,6):
            x=30+k*150
            box=(x,y,x+140,y+140)
            temp=im.crop(box)
            im2=imgproc.normalize(temp)
            im2.save(name+str(i)+str(j*6+k)+"n"+ext)
            

'''
name="testData/"+"test"
ext=".jpg"
imgname=name+ext
im=Image.open(imgname)
nimg=imgproc.normalize(im)
nimg.save(name+"n"+ext)
print "DONE!"
'''       
