"""Render recorded valid points over the supplied target image. No model comparison."""
import argparse
import csv
from pathlib import Path


def render(csv_path,image,output,show=False):
    import numpy as np
    import matplotlib
    if not show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(csv_path,encoding='utf-8-sig',newline='') as handle:
        rows=[r for r in csv.DictReader(handle) if r['valid']=='1']
    points=np.array([[float(r['gaze_x_image_px']),float(r['gaze_y_image_px'])] for r in rows])
    if not len(points):return None
    good=points[(points[:,0]>=0)&(points[:,0]<1920)&(points[:,1]>=0)&(points[:,1]<1080)]
    counts,_,_=np.histogram2d(good[:,1] if len(good) else [],good[:,0] if len(good) else [],
                             bins=(108,192),range=((0,1080),(0,1920)))
    # Small separable Gaussian in pure NumPy, so no additional SciPy installation is needed.
    x=np.arange(-4,5);kernel=np.exp(-x*x/(2*1.5**2));kernel/=kernel.sum()
    density=np.apply_along_axis(lambda a:np.convolve(a,kernel,mode='same'),0,counts)
    density=np.apply_along_axis(lambda a:np.convolve(a,kernel,mode='same'),1,density)
    alpha=np.zeros_like(density) if density.max()==0 else np.clip(density/density.max(),0,1)*.78
    fig,ax=plt.subplots(figsize=(16,9),layout='constrained')
    ax.imshow(plt.imread(image),extent=(0,1920,1080,0))
    ax.imshow(density,extent=(0,1920,1080,0),cmap='turbo',alpha=alpha,interpolation='bilinear')
    ax.set(xlim=(0,1920),ylim=(1080,0),xlabel='Reference image x (px)',ylabel='Reference image y (px)')
    ax.set_title(f'{rows[0]["model"]}: manually recorded gaze sample density | valid {len(rows)}, in-image {len(good)}')
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180)
    if show:plt.show()
    plt.close(fig)
    return Path(output)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--csv',required=True);p.add_argument('--image',required=True)
    p.add_argument('--output',required=True);p.add_argument('--show',action='store_true');args=p.parse_args()
    render(args.csv,args.image,args.output,args.show)
