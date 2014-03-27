from PIL import Image
import numpy

#****************************
n_class=2
n_rows=4       #each row has 6 samples

#*****************************

def normalize(im):
    rows=im.size[1]
    cols=im.size[0]
    #print rows,cols
    a=numpy.array(im.convert('L'))

    '''
    rowno=0
    for x in a:
        print rowno
        print x
        rowno+=1
    '''
    ymin=0
    xmin=0
    ymax=rows
    xmax=cols
    flag=0
    for y in range(0,rows):
        for x in range(0,cols):
            if(a[y][x] < 40):
                ymin=y
                flag=1
                break
        if(flag):
            break
    flag=0        
    for x in range(0,cols):
        for y in range(ymin,rows):
            if(a[y][x] < 40):
                xmin=x
                flag=1
                break
        if(flag):
            break
    flag=0
    for y in range(ymin,rows):
        flag=0
        for x in range(xmin,cols):
            if( a[y][x] < 40):
                flag=1
                break
        if(not(flag)):
            ymax=y
            break

    flag=0
    for x in range(xmin,cols):
        flag=0
        for y in range(ymin,rows):
            if(a[y][x] < 40):
                flag=1
                break
        if(not(flag)):
            xmax=x
            break



    #print ymin,xmin
    #print ymax,xmax

    mx=(xmax-xmin)/20     #margin
    my=(ymax-ymin)/20 
    xmin-=mx
    if(xmin<0):
        xmin=0
    ymin-=my
    if(ymin<0):
        ymin=0
    xmax+=mx
    if(xmax>cols):
        xmax=cols
    ymax+=my
    if(ymax>rows):
        ymax=rows
    box=(xmin,ymin,xmax,ymax)
    region=im.crop(box)
    #region=region.resize((64,62))
    im2=Image.fromarray(numpy.uint8(region.convert('L')))
    #im2.save("result.jpg")
    return im2


name="trainData/"
ext=".jpg"
for i in range(0,n_class):            #class no
    imagename=name+str(i)+ext
    im=Image.open(imagename)
    
    for j in range(0,n_rows):        #no. of samples(rows,cols)
        y=60+j*150
        for k in range(0,6):
            x=30+k*150
            box=(x,y,x+140,y+140)
            temp=im.crop(box)
            im2=normalize(temp)
            im2.save(name+str(i)+str(j*6+k)+"n"+ext)
            
        
#print "DONE!"

          
