from PIL import Image
import numpy
from sklearn import svm
#from sklearn import preprocessing as preproc

#******************************************

n_class=10
n_samples=24        #no. of test samples

n_testfiles=2
n_testcases=24     #6x no_of_rows( testcases per file)

#******************************************

def isblack(pixel):
    if( pixel < 130):
        return 1
    else:
        return 0
    
def blackarea(a,y1,y2,x1,x2):
    black=0
    for y in range(y1,y2):
        for x in range(x1,x2):
            if( isblack(a[y][x]) ):
                black+=1
    blackper=float(black)*100/float(((y2-y1)*(x2-x1)))
    #print black,(y2-y1)*(x2-x1)
    #print blackper
    return blackper

def hmean(a,y1,y2,x1,x2):
    rowmean=[]
    for y in range(y1,y2):
        n=0
        sum=0
        mean=(x2-x1)/2
        for x in range(x1,x2):
            if( isblack(a[y][x]) ):
                sum=sum+x
                n+=1
        #print "count:",n
        if(n!=0):
            mean=float(sum)/float(n)
        rowmean.append(mean)
    #print "rowmean:",rowmean
    hcenter=(x2-x1)/2
    sum=0
    mean=hcenter
    for i in range(0,len(rowmean)):
        sum+=rowmean[i]
    if(len(rowmean)!=0):
        mean=float(sum)/float(len(rowmean))
    #print "overallmean:",mean
    #print "hcenter:",hcenter
    dist=abs(mean-hcenter)

    ratio=float(dist)*100/float(hcenter)

    #print "ratio:",ratio
    return ratio

def vmean(a,y1,y2,x1,x2):
    colmean=[]
    for x in range(x1,x2):
        n=0
        sum=0
        mean=(y2-y1)/2
        for y in range(y1,y2):
            if( isblack(a[y][x]) ):
                sum=sum+y
                n+=1
        #print "count:",n
        if(n!=0):
            mean=float(sum)/float(n)
        colmean.append(mean)
    #print "rowmean:",rowmean
    vcenter=(y2-y1)/2
    sum=0
    mean=vcenter
    for i in range(0,len(colmean)):
        sum+=colmean[i]
    if(len(colmean)!=0):
        mean=float(sum)/float(len(colmean))
    #print "overallmean:",mean
    #print "hcenter:",hcenter
    dist=abs(mean-vcenter)

    ratio=float(dist)*100/float(vcenter)

    #print "ratio:",ratio
    return ratio
    
def getfeat(imgname):
    img=Image.open(imgname)
    r=img.size[1]   #rows    == height
    c=img.size[0]   #columns == width

    a=numpy.array(img.convert('L'))

    #init part

    total=blackarea(a,0,r,0,c)            #feat0
        
    z1=blackarea(a,0,(r/3),0,(c/3))       #feat1
    z2=blackarea(a,0,r/3,c/3,2*c/3)       #feat2
    z3=blackarea(a,0,r/3,2*c/3,c)         #feat3
        
    z4=blackarea(a,r/3,2*r/3,0,c/3)       #feat4
    z5=blackarea(a,r/3,2*r/3,c/3,2*c/3)   #feat5
    z6=blackarea(a,r/3,2*r/3,2*c/3,c)     #feat6

    z7=blackarea(a,2*r/3,r,0,c/3)         #feat7
    z8=blackarea(a,2*r/3,r,c/3,2*c/3)     #feat8
    z9=blackarea(a,2*r/3,r,2*c/3,c)       #feat9
    
    r1=blackarea(a,0,r/3,0,c)
    r2=blackarea(a,r/3,2*r/3,0,c)
    r3=blackarea(a,2*r/3,r,0,c)

    c1=blackarea(a,0,r,0,c/3)
    c2=blackarea(a,0,r,c/3,2*c/3)
    c3=blackarea(a,0,r,2*c/3,c)
    
    hm=hmean(a,0,r,0,c)
    vm=vmean(a,0,r,0,c)

    hm1=hmean(a,0,r/3,0,c)
    hm2=hmean(a,r/3,2*r/3,0,c)
    hm3=hmean(a,2*r/3,r,0,c)

    vm1=vmean(a,0,r,0,c/3)
    vm2=vmean(a,0,r,c/3,2*c/3)
    vm3=vmean(a,0,r,2*c/3,c)
    
    #print temp
    temp=[total,z1,z2,z3,z4,z5,z6,z7,z8,z9,hm,vm,r1,r2,r3,c1,c2,c3,hm1,hm2,hm3,vm1,vm2,vm3]
    #scaler=preproc.StandardScaler().fit(temp)
    #temp=scaler.transform(temp)
    #temp=preproc.scale(temp)
    return temp

from sklearn.externals import joblib
clf=svm.SVC(kernel='linear',cache_size=500)
#clf=svm.LinearSVC()
   
    
try:
    clf=joblib.load('savedata/trainedFile.pkl')    
    print "Using trained model"
except:
    print "Building new model"
    trainData=[]
    trainClass=[]
    for i in range(0,n_class):            #class no
        for j in range(0,n_samples):       #no. of samples
            name="trainData/"+str(i)+str(j)+"n"
            ext=".jpg"
            imgname=name+ext
            temp=[]
            temp=getfeat(imgname)
            
            trainData.append(temp)
            trainClass.append(i)
     
    #print trainData
    #print trainClass

    clf.fit(trainData,trainClass)
    joblib.dump(clf,'savedata/trainedFile.pkl')

print clf
#print clf.fit_status_

print "Support vectors per class:"
print clf.n_support_


#********************** TESTING PART **************

print "Predicted Output:"
k=0
op=[]
oprow=[]
for i in range(0,n_testfiles):                                    #no of test files
    for j in range(0,n_testcases):                                #no of test cases
        name="testData/"+"test"+str(i)+str(j)+"n"        #filename here
        ext=".jpg"
        imgname=name+ext
        temp=[]
        temp=getfeat(imgname)
        #print temp
        if(j%6==0):
            print op
            op=[]
        
        op.append(int(clf.predict(temp)))
        #print clf.support_vectors_
        #e=input()


print op    #last row
#print temp
#print clf.predict_proba(temp)
