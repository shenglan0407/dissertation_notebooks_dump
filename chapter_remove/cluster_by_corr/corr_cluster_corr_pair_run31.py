# cluster and save labels
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import h5py
import os

from loki.utils.postproc_helper import *
from loki.RingData import DiffCorr
from loki.utils import stable

def normalize_shot(ss, this_mask):
    if ss.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        ss = ss.astype(np.float64)
    
    ss *=this_mask
    mean_ss = ss.sum(-1)/this_mask.sum(-1) 
    ss = ss-mean_ss[:,None]
    return np.nan_to_num(ss*this_mask)


def normalize(d):
    x=d.copy()
    x-=x.min()
    return x/(x.max()-x.min())

# output name
fname_output = '/reg/d/psdm/cxi/cxilp6715/results/small_experiments/run31_corrCluster_eigneimgs/pac10_kmeans15_2.h5'
f_out = h5py.File(fname_output,'w')
# load q values
qvalue = np.load('/reg/neh/home/qiaoshen/dissertation_notebooks/qvalues.npy')
# ###### load shots from a run
run_num=31
PT_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/finer_q/'
f_PI=h5py.File(os.path.join(PT_dir,'run%d.tbl'%run_num),'r')

mask = f_PI['polar_mask_binned'].value
mask = (mask==mask.max())
shots = f_PI['polar_imgs'].value[:]

norm_shots = np.zeros_like(shots, dtype=np.float64)
for idx,ss in enumerate(shots):
    if np.nan_to_num(ss).sum<=0:
        continue
    norm_shots[idx]=normalize_shot(ss,mask)
if norm_shots.shape[0]%2>0:
    norm_shots=norm_shots[:-1]

######## load mask and normalize the shots
phi_offset=30
num_phi=norm_shots.shape[-1]
qs = np.linspace(0,1,shots.shape[1])
dc=DiffCorr(mask[None,:,:],qs,0,pre_dif=True)
mask_corr=dc.autocorr()
##### compute single-shot correlations
dc = DiffCorr(norm_shots-norm_shots.mean(0)[None,:,:],
  qs,0,pre_dif=True)
corr = dc.autocorr()
print corr.shape

corr/=mask_corr
corr=corr[:,:,phi_offset:num_phi/2-phi_offset]

diff_shots = norm_shots[::2]-norm_shots[1::2]
dc=DiffCorr(diff_shots,qs,0,pre_dif=True)
no_cluster_ac= (dc.autocorr()/mask_corr).mean(0)
f_out.create_dataset('raw_corrs',data=no_cluster_ac)

####### do PCA on the shots and cluster them with Kmeans
for qidx in range(10, 35):
    print('cluster for qidx %d'%qidx)
    f_out.create_group('q%d'%qidx)
    ####### do PCA on the shots and cluster them with Kmeans
    pca=PCA(n_components=10)

    new_corr=pca.fit_transform(corr[:,qidx,:])
    kmeans=KMeans(n_clusters=15)
    kmeans.fit(new_corr)

    # sort the polar intensities into cluster
    # compute cluster correlations
    all_ac=[]
    num_shots=[]

    ave_cluster_corr = []
    for ll in sorted(np.unique(kmeans.labels_)):


        ss=norm_shots[kmeans.labels_==ll]

        if ss.shape[0]<2:
            continue
        if ss.shape[0]%2>0:
            ss = ss[:-1]

        f_out.create_dataset('q%d/norm_shots_%d'%(qidx,ll),data=ss[:,qidx])

        this_mask=mask.copy()
        # mask correlations
        dc=DiffCorr(this_mask[None,:,:],qs,0,pre_dif=True)
        mask_ac=dc.autocorr()
        
        # difference correlations of the cluster
        ss_diff = ss[::2]-ss[1::2]
        dc=DiffCorr(ss_diff,qs,0,pre_dif=True)
        ac=dc.autocorr()/mask_ac
        
        all_ac.append(ac[:,qidx])
        num_shots.append(ss.shape[0])
        
        ave_cluster_corr.append(ac.mean(0)[qidx])
        
    combined_ac=np.concatenate(all_ac).mean(0)


    # compute and store asymmetries
    cluster_cor_asym =[] 
    for ii ,cc in enumerate(ave_cluster_corr):
        nc=normalize(cc[phi_offset:num_phi/2-phi_offset])
        cluster_cor_asym.append( (np.abs(nc-nc[::-1])).mean() )
    cluster_cor_asym=np.array(cluster_cor_asym)
    ave_cluster_corr=np.array(ave_cluster_corr)

    f_out.create_dataset('q%d/asym'%qidx, data=cluster_cor_asym)
    f_out.create_dataset('q%d/ave_clus_cor'%qidx, data=ave_cluster_corr)
    f_out.create_dataset('q%d/clus_nshots'%qidx,data = np.array(num_shots) )
    f_out.create_dataset('q%d/ave_cor'%qidx,data = combined_ac )

f_out.close()
print('Done!')

