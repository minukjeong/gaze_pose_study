"""Image-plane head/shoulder protractor, not anatomical cervical angles."""
import cv2
import numpy as np


def measure(points, scores, baseline=None, threshold=.4):
    if points is None or scores is None:
        return None
    p,s=np.asarray(points,float),np.asarray(scores,float)
    if p.ndim!=2 or p.shape[0]<7 or p.shape[1]!=2 or s.shape[0]<7:
        return None
    valid=np.isfinite(p).all(1)&np.isfinite(s)&(s>=threshold)
    if not valid[[0,1,2,5,6]].all():
        return None
    center=(p[5]+p[6])/2
    tangent=p[6]-p[5]
    width=np.linalg.norm(tangent)
    if width<10:
        return None
    if tangent[0]<0:
        tangent=-tangent
    tangent/=width
    normal=np.array([tangent[1],-tangent[0]])
    # Fixed head landmark definition avoids switching between nose and ears.
    eyes=(p[1]+p[2])/2
    head=(eyes+p[0])/2
    vector=head-center
    if np.linalg.norm(vector)<5:
        return None
    lateral=np.degrees(np.arctan2(vector@tangent,vector@normal))
    eye_width=np.linalg.norm(p[1]-p[2])
    if eye_width<3:
        return None
    pitch=np.degrees(np.arctan2(p[0,1]-eyes[1],eye_width))
    ref=None if baseline is None else baseline.get('face_pitch_proxy')
    pitch_change=None if ref is None or not np.isfinite(ref) else float(pitch-ref)
    return dict(center=center,head=head,tangent=tangent,normal=normal,width=width,
                points=p.copy(),valid=valid.copy(),
                lateral=float(lateral),pitch_change=pitch_change,
                shoulder=float(np.degrees(np.arctan2(tangent[1],tangent[0]))))


def dial(image, center, normal, tangent, radius, angle, color=(0,210,255)):
    def point(a,r=radius):
        a=np.radians(a)
        return np.round(center+r*(np.cos(a)*normal+np.sin(a)*tangent)).astype(int)
    angles=np.linspace(-90,90,73)
    arc=np.array([point(a) for a in angles])
    cv2.polylines(image,[arc],False,(175,175,175),1,cv2.LINE_AA)
    for a in range(-90,91,10):
        cv2.line(image,tuple(point(a)),tuple(point(a,radius-(9 if a%30==0 else 4))),
                 (200,200,200),1,cv2.LINE_AA)
    c=tuple(np.round(center).astype(int))
    cv2.line(image,c,tuple(point(0,radius+12)),(80,210,90),2,cv2.LINE_AA)
    if angle is not None:
        # Needle shows the angle without clipping the numerical measurement.
        cv2.line(image,c,tuple(point(angle,radius+10)),color,3,cv2.LINE_AA)
        swept=np.array([point(a,radius*.65) for a in np.linspace(0,angle,max(2,int(abs(angle))+1))])
        cv2.polylines(image,[swept],False,color,3,cv2.LINE_AA)
    cv2.circle(image,c,4,(255,255,255),-1,cv2.LINE_AA)


def interior_angle(a, vertex, b):
    u,w=np.asarray(a)-vertex,np.asarray(b)-vertex
    lengths=np.linalg.norm(u)*np.linalg.norm(w)
    if lengths<1e-6:
        return None
    return float(np.degrees(np.arccos(np.clip(u@w/lengths,-1.,1.))))


def joint_arc(image, a, vertex, b, radius):
    angle=interior_angle(a,vertex,b)
    if angle is None:
        return
    u,w=a-vertex,b-vertex
    start=np.arctan2(u[1],u[0])
    sweep=(np.arctan2(w[1],w[0])-start+np.pi)%(2*np.pi)-np.pi
    radius=min(radius,np.linalg.norm(u)*.45,np.linalg.norm(w)*.45)
    theta=np.linspace(start,start+sweep,max(3,int(angle)+1))
    arc=np.round(vertex+radius*np.c_[np.cos(theta),np.sin(theta)]).astype(int)
    cv2.polylines(image,[arc],False,(0,210,255),3,cv2.LINE_AA)
    mid=start+sweep/2
    anchor=vertex+(radius+22)*np.array([np.cos(mid),np.sin(mid)])
    text=f'{angle:.0f}'
    (tw,th),_=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.7,2)
    h,iw=image.shape[:2]
    xy=(int(np.clip(anchor[0]-tw/2,3,max(3,iw-tw-14))),
        int(np.clip(anchor[1]+th/2,th+3,h-5)))
    cv2.putText(image,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.7,(0,0,0),5,cv2.LINE_AA)
    cv2.putText(image,text,xy,cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2,cv2.LINE_AA)
    degree=(xy[0]+tw+5,xy[1]-th+3)
    cv2.circle(image,degree,4,(0,0,0),3,cv2.LINE_AA)
    cv2.circle(image,degree,3,(255,255,255),1,cv2.LINE_AA)


def draw_camera(image, values):
    if values is None:
        return image
    v=values
    center,head=v['center'],v['head']
    radius=float(np.clip(v['width']*.15,18,48))
    a=center-v['tangent']*v['width']/2
    b=center+v['tangent']*v['width']/2
    down=center-v['normal']*v['width']*.5
    def line(x,y,color=(255,255,255)):
        cv2.line(image,tuple(np.round(x).astype(int)),tuple(np.round(y).astype(int)),color,2,cv2.LINE_AA)
    line(a,b,(200,210,65))
    line(center,head)
    # Dashed downward axis is a virtual shoulder-normal, not a detected torso.
    for t in np.arange(0,1,.16):
        line(center+t*(down-center),center+min(t+.09,1)*(down-center))
    for point in (a,b,head):
        cv2.circle(image,tuple(point.astype(int)),4,(255,255,255),-1,cv2.LINE_AA)
    cv2.circle(image,tuple(center.astype(int)),5,(255,255,255),1,cv2.LINE_AA)
    joint_arc(image,head,center,down,radius)
    p=v.get('points'); valid=v.get('valid')
    if p is not None:
        # Show real shoulder/elbow joint angles only when the arm is visible.
        for sh,el,wr in ((5,7,9),(6,8,10)):
            if el<len(p) and valid[sh] and valid[el]:
                line(p[sh],p[el])
                joint_arc(image,center,p[sh],p[el],radius*.75)
                cv2.circle(image,tuple(p[el].astype(int)),4,(255,255,255),-1,cv2.LINE_AA)
                if wr<len(p) and valid[wr]:
                    line(p[el],p[wr])
                    joint_arc(image,p[sh],p[el],p[wr],radius*.65)
    cv2.putText(image,'2D angles | dashed axis: virtual shoulder reference',(8,image.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX,.4,(0,0,0),3,cv2.LINE_AA)
    cv2.putText(image,'2D angles | dashed axis: virtual shoulder reference',(8,image.shape[0]-12),
                cv2.FONT_HERSHEY_SIMPLEX,.4,(255,255,255),1,cv2.LINE_AA)
    return image


def draw_panel(panel, values):
    cv2.putText(panel,'HEAD / SHOULDER PROTRACTOR',(18,58),cv2.FONT_HERSHEY_SIMPLEX,
                .55,(55,55,55),1,cv2.LINE_AA)
    for cx,label,angle in ((145,'LATERAL / 2D',None if values is None else values['lateral']),
                           (420,'BOW CHANGE / PROXY',None if values is None else values['pitch_change'])):
        dial(panel,np.array([cx,190.]),np.array([0.,-1.]),np.array([1.,0.]),65,angle)
        shown='N/A' if angle is None else f'{angle:+.1f} deg'
        cv2.putText(panel,label,(cx-105,95),cv2.FONT_HERSHEY_SIMPLEX,.43,(55,55,55),1,cv2.LINE_AA)
        cv2.putText(panel,shown,(cx-55,222),cv2.FONT_HERSHEY_SIMPLEX,.7,(140,45,90),2,cv2.LINE_AA)
    text='Shoulder normal = zero; image-right is positive'
    if values is not None:
        text+=f" | shoulder {values['shoulder']:+.1f}"
    cv2.putText(panel,text,(15,250),cv2.FONT_HERSHEY_SIMPLEX,.36,(70,70,70),1,cv2.LINE_AA)
    cv2.putText(panel,'Bow proxy needs reference; not cervical flexion',(15,268),
                cv2.FONT_HERSHEY_SIMPLEX,.4,(50,50,170),1,cv2.LINE_AA)
    return panel
