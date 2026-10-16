import os, sys;sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from racing.track import generate_track
plt.rcParams["font.family"]=["DejaVu Sans","Noto Sans Thai"]
def corners(t,thr=150):
    c=t["curvature_radius"]<thr;n=len(c)
    # count rising edges circularly
    return int(np.sum(c & ~np.roll(c,1)))
def draw(ax,t,title):
    L,R=t["left"],t["right"]
    ax.add_patch(Polygon(np.vstack([L,L[:1]]),closed=True,fc="#2b2b2b",ec="none",zorder=1))
    ax.add_patch(Polygon(np.vstack([R,R[:1]]),closed=True,fc="#eef2ee",ec="none",zorder=2))
    # orientation: if right poly is the outer one, swap fill
    def area(p):return 0.5*np.sum(p[:,0]*np.roll(p[:,1],-1)-np.roll(p[:,0],-1)*p[:,1])
    if abs(area(R))>abs(area(L)):
        ax.patches[-1].remove();ax.patches[-1].remove()
        ax.add_patch(Polygon(R,closed=True,fc="#2b2b2b",ec="none",zorder=1))
        ax.add_patch(Polygon(L,closed=True,fc="#eef2ee",ec="none",zorder=2))
    for W in (L,R):
        ax.plot(*np.vstack([W,W[:1]]).T,color="white",lw=0.6,zorder=3)
    # kerbs in tight corners
    tight=t["curvature_radius"]<80
    for W in (L,R):
        for i in np.where(tight)[0]:
            j=(i+1)%len(W)
            ax.plot([W[i,0],W[j,0]],[W[i,1],W[j,1]],color="#d62828" if (i//2)%2 else "white",lw=2.2,solid_capstyle="butt",zorder=4)
    # start line
    n0=t["normals"][0];c0=t["centerline"][0];hw=t["width"]/2
    ax.plot([c0[0]-n0[0]*hw,c0[0]+n0[0]*hw],[c0[1]-n0[1]*hw,c0[1]+n0[1]*hw],color="#f4c20d",lw=3,zorder=5)
    tg=t["tangents"][0]
    ax.annotate("",xy=c0+tg*120,xytext=c0+tg*20,arrowprops=dict(arrowstyle="->",color="#f4c20d",lw=1.5),zorder=6)
    ax.set_aspect("equal");ax.axis("off");ax.autoscale_view()
    ax.set_title(title,fontsize=9,color="#222")
def gallery(diff,seeds,fn,head):
    fig,axs=plt.subplots(2,4,figsize=(16,8.4),facecolor="#eef2ee")
    for ax,s in zip(axs.flat,seeds):
        t=generate_track(seed=s,difficulty=diff)
        draw(ax,t,f"seed {s}  |  {t['length']/1000:.2f} km  |  {corners(t)} corners  |  min R {t['curvature_radius'].min():.0f} m")
    fig.suptitle(head,fontsize=14,color="#111")
    fig.tight_layout();fig.savefig(fn,dpi=110,facecolor=fig.get_facecolor());plt.close(fig)
gallery(1.0,range(8),"figures/tracks_d10.png","Difficulty 1.0  (width 15 m, F1)")
gallery(0.5,range(8),"figures/tracks_d05.png","Difficulty 0.5  (width 22.5 m)")
gallery(0.0,range(8),"figures/tracks_d00.png","Difficulty 0.0  (width 30 m, curriculum start)")
# hero: one hard circuit, large
t=generate_track(seed=6,difficulty=1.0)
fig,ax=plt.subplots(figsize=(12,8),facecolor="#eef2ee")
draw(ax,t,f"seed 6  |  {t['length']/1000:.2f} km  |  {corners(t)} corners  |  width 15 m  |  tightest radius {t['curvature_radius'].min():.0f} m")
fig.tight_layout();fig.savefig("figures/track_hero.png",dpi=160,facecolor=fig.get_facecolor());plt.close(fig)
