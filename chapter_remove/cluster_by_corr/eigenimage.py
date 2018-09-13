
# take a run, do PCA at each q

# save the shots into train and test, save the 
import h5py
import os

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

import numpy as np

import sys

from loki.RingData.DiffCorr import DiffCorr

def normalize_shot(ss, this_mask):
    if ss.dtype != 'float64':
        # shots need to be float64 or more. 
        # float32 resulted in quite a bit of numerical error 
        ss = ss.astype(np.float64)
    
    ss *=this_mask
    mean_ss = ss.sum(-1)/this_mask.sum(-1) 
    ss = ss-mean_ss[:,None]
    return np.nan_to_num(ss*this_mask)

def reshape_unmasked_values_to_shots(shots,mask):
    # this takes vectors of unmasked values, and reshaped this into their masked forms
    # mask is 2D, shots are 1D
    assert(shots.shape[-1]==np.sum(mask) )
    flat_mask = mask.flatten()
    reshaped_shots = np.zeros( (shots.shape[0],mask.size), dtype=shots.dtype)
    
    reshaped_shots[:, flat_mask==1] = shots
    
    return reshaped_shots.reshape((shots.shape[0],mask.shape[0],mask.shape[1]))
    

# load the shots
finput_name = '/reg/d/psdm/cxi/cxilp6715/results/small_experiments/run31_corrCluster_eigneimgs/pca10_kmeans15.h5'
f_PI=h5py.File(finput_name,'r')
# output name
fname_output='/reg/d/psdm/cxi/cxilp6715/results/small_experiments/run31_corrCluster_eigneimgs/eigenimgs.h5'
f_out = h5py.File(fname_output,'w')
# load q values
qvalue = np.load('/reg/neh/home/qiaoshen/dissertation_notebooks/qvalues.npy')
# ###### load mask
PT_dir = '/reg/d/psdm/cxi/cxilp6715/scratch/combined_tables/finer_q/'
f=h5py.File(os.path.join(PT_dir,'run31.tbl'),'r')
mask = np.array(f['polar_mask_binned'].value==f['polar_mask_binned'].value.max(), dtype = int)

##### range of q inds to make eigne images for
qcluster_inds=range(10,35)

###max number of eigen images to substract
max_pca_limit=15
#if True, save to get error bars
save_cors=False

for qidx in qcluster_inds:
    print('PCA denoising for qidx %d'%qidx)
    q_group = 'q%d'%qidx
    if q_group not in f_out.keys():
        f_out.create_group(q_group)
    # mask and dummy q value
    partial_mask=mask[qidx][None,:]
    qvalues = np.linspace(0,1,partial_mask.shape[0])
    # kmeans clusters
    kmeans_clus_keys=[kk for kk in f_PI[q_group].keys() if kk.startswith('norm_shots')]

    # find eigenimages for each cluster
    for kk in kmeans_clus_keys:
        cluster_group='q%d/k%s'%(qidx,kk.split('_')[-1])
        f_out.create_group( cluster_group)
        # train shots for the cluster
        norm_shots = f_PI[q_group][kk].value[:,None,:]
        Train = norm_shots[:, partial_mask==1]
        num_shots = norm_shots.shape[0]
    
        print ("%d shots"%(num_shots))
        
        max_pca = np.min((num_shots-1,max_pca_limit))

        print('denoisng with PCA critical num_pca_components = %d...'%max_pca)
        if 'pca_components' not in f_out[cluster_group].keys():
            # if there is no pca component saved, then run it and save the components
            pca=PCA(n_components=np.min((50,num_shots-1)), whiten = False)

            new_Train=pca.fit_transform(Train)
            if 'explained_variance_ratio' not in f_out[cluster_group].keys():
                f_out.create_dataset('%s/explained_variance_ratio'%cluster_group,data=pca.explained_variance_ratio_)

            # get back the masked images and components
            components=pca.components_
            f_out.create_dataset('%s/pca_components'%cluster_group,data=components)
        else:
            #load the components previously save the then do the transformations
            components = f_out['%s/pca_components'%cluster_group].value
            _m = Train.astype(np.float64).mean(0)
            new_Train = (Train.astype(np.float64)-_m).dot(components.T)

        # get back the masked images and components
     
        masked_mean_train =reshape_unmasked_values_to_shots(Train,partial_mask).mean(0)
        
        #### this is just for saving to get error bars
        if save_cors:
            grp=f_out['q%d'%qidx]
            nn=grp['num_pca_cutoff'].value
            if 'all_difcors' in grp['pca%d'%nn].keys():
                print('already save dif cors for this cutoff (%d) at q%d'%(nn,qidx))
            else:

                Train_noise = new_Train[:,:nn].dot(components[:nn])
                denoise_Train= reshape_unmasked_values_to_shots(norm_shots-Train_noise-Train.mean(0)[None,:]
                                                            , partial_mask)

                dc=DiffCorr(denoise_Train,qvalues,0,pre_dif=False)
                Train_difcor= dc.autocorr()

                f_out.create_dataset('%s/pca%d/all_train_difcors'%(cluster_group,nn)
                    ,data=Train_difcor)
            del norm_shots
            continue
        

        # denoise
        for nn in range(1, max_pca):
            pca_group = '%s/pca%d'%(cluster_group,nn)
            if 'pca%d'%nn in f_out[cluster_group].keys():
                print("pca denoise at pca n_components = %d is already done. Skip!"%nn)
                continue

            if nn>0:

                print('subtracting noise with pca n_components =  %d'%nn)
                Train_noise = new_Train[:,:nn].dot(components[:nn])
                denoise_Train= reshape_unmasked_values_to_shots(Train-Train_noise-Train.mean(0)[None,:]
                                                            , partial_mask)

                
                dc=DiffCorr(denoise_Train,qvalues,0,pre_dif=False)
                Train_difcor= dc.autocorr()
                
                dc=DiffCorr(partial_mask,qvalues,0,pre_dif=True)
                mask_ac= dc.autocorr()
                Train_difcor/=mask_ac

                f_out.create_dataset(pca_group
                    ,data=Train_difcor.mean(0))
    
        
        if 'num_shots' not in f_out[cluster_group].keys():        
            f_out.create_dataset('%s/num_shots'%cluster_group, data=norm_shots.shape[0])

        del norm_shots

print ("done!")
f_out.close()