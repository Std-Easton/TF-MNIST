from demo_classiffier import *

num9 ='num_9.png'
num5 ='num_5.png'
num4 ='num_4.png'
num7 ='num_7-2.png'
num8 ='num_8.png'
name =num9
thickness =1.0
angle =0.0
scale =1.0
axcolor = 'lightgoldenrodyellow'

#fig, ax = plt.subplots()
fig =plt.figure(1,(10,10))
ax1 =plt.subplot2grid((5,5),(0,1),colspan =4)
ax2 =plt.subplot2grid((5,5),(1,0),rowspan =3,colspan =5)

## main plot area
def refresh(elastic =False):
    global name
    global thickness
    global angle
    global scale
    #plt.axes([0.3, 0.25, 0.5, 0.5])
    #image = cv2.cvtColor(cv2.imread(name), cv2.COLOR_BGR2GRAY)
    image =imageprepare('pics/'+name).reshape(28, 28)
    if elastic:
        image =elastic_transform(image)
    image =thicken(image, thickness)
    image =rotate(image, angle)
##    image =zoom(image, scale)
##    image =transformation(image, vector)
    ax2.imshow(image,'gray')
    predint = predictint(image.reshape(784), "model2-1.ckpt")

##    print 'predicted number: ',(np.argmax(predint)) #first value in list
##    print 'confidence rate:' ,max(predint)
##    num =int(0)
##    for i in predint:
##	print num,':',round(i,3)
##	num+=1
    width =0.5
    ax1.bar(np.arange(10),np.ones(10), width, linewidth =0,color='white')
    ax1.bar(np.arange(10),predint, width, linewidth =0,color='r')
    ax1.set_xticks(np.arange(10) + width/2)
    ax1.set_xticklabels('0123456789')
    ax1.set_title('predicted number is: '+ str(np.argmax(predint)),fontsize =25 )
refresh()

## slider bar
axthick = plt.axes([0.25, 0.1, 0.65, 0.03], axisbg=axcolor)
axangle = plt.axes([0.25, 0.15, 0.65, 0.03], axisbg=axcolor)

sthick = Slider(axthick, 'thickness', 0.1, 3, valinit=1.0)
sangle = Slider(axangle, 'angle', -90.0, 90.0, valinit=0.0)

def update(val):
    global thickness
    global angle
    angle = sangle.val
    thickness = sthick.val
    refresh()
    fig.canvas.draw_idle()
sthick.on_changed(update)
sangle.on_changed(update)

## button 
resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
elasticax =plt.axes([0.5, 0.025, 0.1, 0.04])
reset_button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')
elastic_button = Button(elasticax, 'elastic', color=axcolor, hovercolor='0.975')

def reset(event):
    sthick.reset()
    sangle.reset()
reset_button.on_clicked(reset)

def elastic(event):
##    sthick.reset()
##    sangle.reset()
    refresh(elastic =True)
    fig.canvas.draw_idle()
elastic_button.on_clicked(elastic)    

## radio 
rax = plt.axes([0.025, 0.5, 0.15, 0.15], axisbg=axcolor)
radio = RadioButtons(rax, (num9, num5, num4, num7, num8), active=0)

def picSelect(label):
    global name
    name =label
    refresh()
    fig.canvas.draw_idle()
radio.on_clicked(picSelect)


plt.show()

