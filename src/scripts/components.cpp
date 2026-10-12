// SPDX-License-Identifier: MIT-0
// Exact 4-connected component selection; tie break is first row-major cell.
#include <cstdint>
#include <algorithm>
extern "C" int select_component(const uint8_t* a,int h,int w,int bg,int same,
                                int smallest,int crop,uint8_t* out,int* dims) {
    int labels[900]={},queue[900],sizes[901]={},first[901]={};
    int count=0;
    for(int start=0;start<h*w;++start) {
        if(a[start]==bg || labels[start])continue;
        int label=++count,head=0,tail=1;
        queue[0]=start;labels[start]=label;first[label]=start;
        while(head<tail) {
            int i=queue[head++],y=i/w,x=i%w;
            int ns[4]={y?i-w:-1,y+1<h?i+w:-1,x?i-1:-1,x+1<w?i+1:-1};
            for(int q:ns)if(q>=0 && !labels[q] && a[q]!=bg && (!same || a[q]==a[start])) {
                labels[q]=label;queue[tail++]=q;
            }
        }
        sizes[label]=tail;
    }
    if(!count)return 0;
    int best=1;
    for(int c=2;c<=count;++c)if((smallest && sizes[c]<sizes[best]) || (!smallest && sizes[c]>sizes[best]))best=c;
    int ymin=0,ymax=h-1,xmin=0,xmax=w-1;
    if(crop) {
        ymin=h;xmin=w;ymax=xmax=-1;
        for(int i=0;i<h*w;++i)if(labels[i]==best) {
            ymin=std::min(ymin,i/w);ymax=std::max(ymax,i/w);
            xmin=std::min(xmin,i%w);xmax=std::max(xmax,i%w);
        }
    }
    dims[0]=ymax-ymin+1;dims[1]=xmax-xmin+1;
    for(int y=ymin;y<=ymax;++y)for(int x=xmin;x<=xmax;++x) {
        int i=y*w+x,j=(y-ymin)*dims[1]+x-xmin;
        out[j]=labels[i]==best?a[i]:bg;
    }
    return 1;
}
