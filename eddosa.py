import ddosa
import traceback
import ogip
import re
from ddosa import dataanalysis, \
                   ScWData, \
                   DataAnalysis, \
                   GetEcorrCalDB, \
                   remove_withtemplate, \
                   construct_gnrl_scwg_grp, \
                   heatool, \
                   DataFile, \
                   RevForScW, \
                   ScWData, \
                   BasicEventProcessingSummary, \
                   SpectraBins, \
                   ii_spectra_extract


import sys
import os
import pprint
import gzip, glob
import astropy.io.fits
import pandas as pd

#import ltdata

import fit_ng
import pilton

from copy import deepcopy
from scipy import ndimage,interpolate

import dataanalysis as da
        
import astropy.io.fits as pyfits
import numpy as np # transition to this...
from numpy import sqrt, array, copy, random
        
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.filters import gaussian_filter1d as g1d
from scipy import stats
from scipy.interpolate import interp1d as i1d
from scipy.interpolate import interp1d,UnivariateSpline

        
import gzip
import dataanalysis.core as da
from functools import reduce

#ddosa.dataanalysis.LogStream(None,lambda x:any([y in x for y in ['heatool','top']]))
ddosa.dataanalysis.printhook.LogStream(None,lambda x:True)
ddosa.dataanalysis.printhook.LogStream("alllog.txt",lambda x:True)

import plot

plot.showg=False

cache_local=dataanalysis.caches.cache_core.Cache()
#cache_local=dataanalysis.Cache

from bcolors import render

class Fit1DSpectrumRev(da.DataAnalysis): pass

class GetLUT2new(ddosa.DataAnalysis):
    input="LUT2_0.0.0"

    def main(self):
        self.datafile=""

class Revolution(ddosa.DataAnalysis):
    #input_scw=ddosa.ScWData

    version="forscw"

    allow_alias=False
    run_for_hashe=True

    def main(self):
        return ddosa.RevForScW(input_scw=ScWData)
    
class ImagingConfig(ddosa.ImagingConfig):
    input_name="standard1"

    def main(self):
        ddosa.ImagingConfig.main(self)
        self.MinCatSouSnr=4
        self.MinNewSouSnr=5
        self.DoPart2=1

class ImageBins(ddosa.DataAnalysis):
    input_name="std4bins"

    def main(self):
        self.bins=[(20,40),(40,80),(80,150),(150,300)]

class CatForSpectra(ddosa.DataAnalysis):
    input_imaging=ddosa.ii_skyimage

    version="v3"
    def main(self):
        if hasattr(self.input_imaging,'empty_results'):
            self.empty_results=True
            return

        catfn="cat4spectra.fits"

        f=pyfits.open(self.input_imaging.srclres.path)
        f[1].data=f[1].data[f[1].data['DETSIG']>10]
        f.writeto(catfn,overwrite=True)

        self.cat=ddosa.DataFile(catfn)

class ibis_isgr_energy(DataAnalysis):
    cached=False

    input_scw=ScWData()
    input_ecorrdata=GetEcorrCalDB

    version="v5_extras"

    def main(self):

        remove_withtemplate("isgri_events_corrected.fits(ISGR-EVTS-COR.tpl)")

        construct_gnrl_scwg_grp(self.input_scw,[\
           # self.input_scw.scwpath+"/isgri_events.fits[ISGR-EVTS-ALL]", \
            self.input_scw.scwpath+"/isgri_events.fits[ISGR-EVTS-ALL]" if not hasattr(self,'input_eventfile') else self.input_eventfile.evts.get_path(),
            self.input_scw.scwpath+"/ibis_hk.fits[IBIS-DPE.-CNV]" \
        ])

        bin=os.environ['COMMON_INTEGRAL_SOFTDIR']+"/spectral/ibis_isgr_energy/ibis_isgr_energy_pha2_optdrift/ibis_isgr_energy"
        ht=heatool(bin)
        ht['inGRP']="og.fits"
        ht['outCorEvts']="isgri_events_corrected.fits(ISGR-EVTS-COR.tpl)"
        ht['useGTI']="n"
        ht['randSeed']=500
        ht['riseDOL']=self.input_ecorrdata.risedol
        ht['GODOL']=self.input_ecorrdata.godol
        ht['supGDOL']=self.input_ecorrdata.supgdol
        ht['supODOL']=self.input_ecorrdata.supodol
        ht['chatter']="4"
        ht.run()


        self.output_events=DataFile("isgri_events_corrected.fits")
        self.events=DataFile("isgri_events_corrected.fits")

class ISGRIEvents(ddosa.ISGRIEvents):
    cached=True
    cache=cache_local

    read_caches=[cache_local.__class__]
    write_caches=[cache_local.__class__]

class RawCalDataRev(da.DataAnalysis):
    input_rev=Revolution

    #cached=True

    version="v1"

    def main(self):
        pattern=self.input_rev.revdir+"/raw/isgri_raw_cal*"
        print("pattern:",pattern)
        files=glob.glob(pattern)
    
        inhdus=[pyfits.open(fn)[2] for fn in files]
        nrows_total=sum([h.data.shape[0] for h in inhdus])

        print("total rows:",nrows_total)

        hdu = pyfits.new_table(inhdus[0].columns, nrows=nrows_total)

        for i in range(len(hdu.columns)):
            nrows_counter=0
            for h in inhdus:
                nrows=h.data.shape[0]
                hdu.data.field(i)[nrows_counter:nrows_counter+nrows]=h.data.field(i)
                nrows_counter+=nrows

        fn="isgri_cal_events_merged.fits"
        hdu.writeto(fn,overwrite=True)
        self.events=da.DataFile(fn)

class BinBackgroundSpectrumP3(ddosa.DataAnalysis): pass
                
#class RevForScW(da.DataAnalysis):
#    run_for_hashe=True
#    input_scw=ScWData
    
#    def main(self):
#        return Revolution(input_revid=self.input_scw.input_scwid.handle[:4])

class RevCalDataRevForScW(RawCalDataRev):
    input_rev=RevForScW

#    run_for_hashe=True
#    input_scw=ScWData
#    
#    def main(self):
#        return Revolution(input_revid=self.input_scw.input_scwid.handle[:4])

class InspectRevCalDataInScW(da.DataAnalysis):
    input_data=RevCalDataRevForScW
    
    def main(self):
        print(self.input_data)

class BinBackgroundSpectrum(ddosa.DataAnalysis):
    input_events=ISGRIEvents
    input_scw=ScWData

    version="v11"

    cached=True
    copy_cached_input=False

    #datafile_restore_mode='url_in_object'

    tag=""

    plot=False
    plot_essentials=False
    plot_2d=False
    save_fits=True
    
    def plot_more(self):
        pass

    def verify_content(self):

        if hasattr(self,'nevents') and self.nevents==0:
            print("no events here")
            return True

        if not hasattr(self,'h1_pha1'):
            print("h1_pha1 missing!")
            return

        return True

    save_extra=False

    multi_polycell=False

    def get_version(self):
        v=self.get_signature()+"."+self.version
        if self.save_extra:
            return v+".extra"
        if self.multi_polycell:
            return v+".mpoly"
        return v

    def main(self):

        bins=(np.arange(2049),np.arange(257))

        pf=pyfits.open(self.input_events.events.get_cached_path())
        evts=pf[1].data

        rawfn=self.input_scw.get_isgri_events()
        rawevts=pyfits.open(rawfn)['ISGR-EVTS-ALL'].data
        
        if evts.shape[0]==0:
            print("no events here")
            self.nevents=0
            return

        if evts.shape[0]!=rawevts.shape[0]:
            print("unequal number of events in "+repr(self.input_scw)+"\n%s\n%s\n: suspecting much BTI; %i vs %i"%(rawfn,self.input_events.events.get_cached_path(),evts.shape[0],rawevts.shape[0]))
            rawevts=None

        self.nevents=evts.shape[0]

        #for a,b in [('ISGRI_ENERGY','ISGRI_PI')]: #,('ISGRI_PHA2','ISGRI_PI'),('ISGRI_PHA1','ISGRI_RT1')]:
        #for a,b,na,nb in [('ISGRI_ENERGY','ISGRI_PI',1,1),('ISGRI_PHA1','ISGRI_RT1',0.5,1),('ISGRI_PHA1','ISGRI_PI',1,1),('ISGRI_PHA2','ISGRI_PI',1,1)]:
        for a,b,na,nb in [('ISGRI_ENERGY','ISGRI_PI',1,1),('ISGRI_PHA2','ISGRI_PI',1,1),('ISGRI_PHA1','ISGRI_RT1',1,1),('ISGRI_PHA1','ISGRI_RT1',0.5,1),('ISGRI_PHA1','ISGRI_PI',0.5,1),('ISGRI_PHA1','ISGRI_PI',1,1),('ISGRI_PHA','ISGRI_RT1',1,1)]:

            def get_evts(k):
                try:
                    return evts[k]
                except:
                    pass
                try:
                    return rawevts[k]
                except:
                    try:
                        print(rawevts)
                    except:
                        pass
                    raise Exception("not found in events: "+k+" in "+rawfn,self.input_events.events.get_cached_path())

            x=get_evts(a)
            y=get_evts(b)
            h2=np.histogram2d(x*na,y*nb,bins=bins)



            print(a,x)
            print(b,y)
            print("np.histogram:",h2)

            key="h2_%s_%s_%.3lg_%.3lg_%s"%(a,b,na,nb,self.tag)
            np.save(key+".npy",h2)

            setattr(self,key,da.DataFile(key+".npy"))
            
            img_orig=h2[0]
            img=gaussian_filter(h2[0],5)

            if self.plot_2d:
                plot.p.figure()
                levels=np.linspace(0,np.log10(img_orig.max()),100)
                plot.p.contourf(np.log10(np.transpose(img_orig)),levels=levels)
             #   plot.p.contour(np.log10(np.transpose(img)),levels=levels)
                plot.p.xlim([10,600])
                plot.p.ylim([20,130])
                plot.p.colorbar()
                plot.plot(key+".png")
                plot.p.xlim([10,300])
                plot.p.ylim([20,130])

                plot.plot(key+"_le.png")
            
            if self.save_fits:
                pyfits.PrimaryHDU(h2[0]).writeto(key+"_le.fits",overwrite=True)


        for rtmin,rtlim in [(16,116),(16,50),(50,116)]:
            selection=(evts['ISGRI_PI']<rtlim) & (evts['ISGRI_PI']>rtmin)

            rtkey="_rt_%i_%i"%(rtmin,rtlim)

            plot.p.figure()
            pha2=evts['ISGRI_PHA2'].astype(float)
            pha2+=random.rand(pha2.shape[0])
            h1=np.histogram(pha2[selection],bins=np.logspace(1,np.log10(2048),300))
            np.save("h1_PHA2%s.npy"%rtkey,h1)
            plot.p.plot(h1[1][1:],h1[0],label="PHA2")
            self.h1_pha2=ddosa.DataFile("h1_PHA2%s.npy"%rtkey)
            np.savetxt("h1_PHA2%s.txt"%rtkey,np.column_stack((h1[1][1:],h1[0])))

            pha1=evts['ISGRI_PHA1'].astype(float)
            pha1+=random.rand(pha1.shape[0])
            h1=np.histogram(pha1[selection],bins=np.logspace(1,np.log10(2048),300))
            plot.p.plot(h1[1][1:]/2,h1[0],label="PHA1/2")
            np.save("h1_PHA1%s.npy"%rtkey,h1)
            self.h1_pha1=ddosa.DataFile("h1_PHA1%s.npy"%rtkey)
            np.savetxt("h1_PHA1%s.txt"%rtkey,np.column_stack((h1[1][1:],h1[0])))
            
            h1=np.histogram(evts['ISGRI_ENERGY'][selection],bins=np.logspace(1,np.log10(2048),300))
            plot.p.plot(h1[1][1:],h1[0],label="ENERGY")
            fn="h1_ENERGY%s.npy"%rtkey
            np.save(fn,h1)
            np.savetxt("h1_ENERGY%s.txt"%(rtkey),np.column_stack((h1[1][1:],h1[0])))
            setattr(self,'h1_energy%s'%rtkey,ddosa.DataFile(fn))

        # model
            energy=h1[1][1:]
            energy_spectrum=h1[0]

            def normalize_model(energy,spectrum,model_energy,width):
                model_line=np.exp(-((energy-model_energy)/width)**2/2)
                model_line[np.isnan(model_line)]=0
                mask=model_line>(model_line.max()/10.)
                model_line=model_line*spectrum[mask].max()
                return model_line


            plot.p.plot(energy,normalize_model(energy,energy_spectrum,59.5,8),label="model")
            plot.p.plot(energy,normalize_model(energy,energy_spectrum,511.,25),label="model")
            
            if self.multi_polycell:
                coords={'corners':[(0,0),(0,3),(3,0),(3,3)],
                        'center': [(1,1),(1,2),(2,1),(2,2)]
                        }

                for cname,cgr in list(coords.items()):
                    pcid_i=rawevts['ISGRI_Y']%4
                    pcid_j=rawevts['ISGRI_Z']%4
                    pcm=np.zeros_like(pcid_j)==1

                    for pci,pcj in cgr:
                       pcm=pcm | ((pcid_i==pci) & (pcid_j==pcj))

                    h1x=np.histogram(evts['ISGRI_ENERGY'][selection & pcm],bins=np.logspace(1,np.log10(2048),300))
                    plot.p.plot(h1x[1][1:],h1x[0]*4.,label="pc "+cname)

            try:
                plot.p.loglog()
            except:
                self.empty_results="no positive values"
                return
            plot.p.ylim([h1[0].max()/50,h1[0].max()*2])
            plot.p.xlim([10,1000])

            plot.p.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


            plot.plot("h1%s_%s.png"%(rtkey,self.tag))
            plot.p.xlim([30,70])
            #plot.plot("h1_lrt%.5lg_%s_le.png"%(rtlim,self.tag))
            plot.p.xlim([400,700])
           # plot.plot("h1_lrt%.5lg_%s_he.png"%(rtlim,self.tag))

        self.plot_more()
            
        if self.save_extra:
            h2=np.histogram2d(evts['ISGRI_ENERGY'],evts['ISGRI_PI'],bins=(np.logspace(1,np.log10(2048),300),np.arange(257)))
            #plot.p.plot(h1[1][1:],h1[0],label="ENERGY")
            fn="h2_energy_pi%s.fits"%("" if self.tag=="" else "_"+self.tag)

            pyfits.HDUList( [pyfits.PrimaryHDU(h2[0]),
                            pyfits.BinTableHDU.from_columns([
                                    pyfits.Column(name='EBOUND', format='E', array=h2[1])
                            ])]).writeto(fn,overwrite=True)
            
            self.h2_energy_pi=da.DataFile(fn)



    def get_h2_pha1_pi(self):
        return np.load(getattr(self,'h2_ISGRI_PHA1_ISGRI_PI_1_1_'+self.tag).open())[0]
    
    def get_h2_energy_pi(self):
        return np.load(getattr(self,'h2_ISGRI_ENERGY_ISGRI_PI_1_1_'+self.tag).open())[0]
    
    def get_h2_energy_pi_300(self):
        return pyfits.open(self.h2_energy_pi.get_path())[0].data

class BiparFile(da.DataAnalysis):
    input_file=None

    def main(self):
        fn=self.input_file.handle

        data=pyfits.open(fn)[0].data

        np.save("h1_pha1.npy",(data.sum(axis=1),np.arange(data.shape[0]+1)/1.))
        self.h1_pha1=da.DataFile("h1_pha1.npy")

        pyfits.PrimaryHDU(data).writeto("h2_pha2_pi.fits",overwrite=True)
        self.h2_pha1_pi=da.DataFile("h2_pha2_pi.fits")

    def get_h2_pha1_pi(self):
        print("opening",self.h2_pha1_pi.path)
        return  pyfits.open(self.h2_pha1_pi.path)[0].data




class BinBackgroundMerged(ddosa.DataAnalysis): pass
        
class Bipar(da.DataAnalysis):
    bipar=BinBackgroundMerged
    allow_alias=False
    run_for_hashe=True

    def main(self):
        print("returning",self.bipar)
        return self.bipar

class FindPeaks(ddosa.DataAnalysis):
    input_histograms=Bipar
    version="v5.1"

    cached=True

    def main(self):

        # no really we should fit it

        line1_energy=59.9
        line2_energy=511.0

        if hasattr(self.input_histograms,'h1_pha1') and False:
            h1,ee=np.load(self.input_histograms.h1_pha1.path)
            ec=(ee[:-1]+ee[1:])/2.
        else:
            h1=pyfits.open(self.input_histograms.h2_pha1_pi.get_path())[0].data.sum(axis=1)
            ee=(np.arange(h1.shape[0]+1))#*0.5
            ec=(ee[:-1]+ee[1:])/2.
            h1=h1*ec

        print(ec)
        print(h1)

        print(ec.shape,ee.shape,h1.shape)

        s=np.logical_and(ec>30*2,ec<80*2)
        emax1=ec[s][(h1/ec)[s].argmax()]
        
        print("PHA1 overall maximum:",emax1)

        print("PHA1 peak:",emax1)

        gain_guess=(emax1+20)/59.

        print("will search for HE peak in",400*gain_guess,570*gain_guess)
            
        s=np.logical_and(ec>200*gain_guess,ec<800*gain_guess)
        s1=np.logical_and(ec>400*gain_guess,ec<550*gain_guess)
        sb=np.logical_and(s,~s1)
        hh=h1[s]
        eh=ec[s]

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(ec[sb]),np.log(h1[sb])) # validate
        hhs=np.exp(np.log(hh)-intercept-np.log(eh)*slope)

        hhs[~s1[s]]=0
        emax2=eh[hhs.argmax()]
        
        print("PHA1 peak at high energy:",emax2)

        p=plot.p
        p.figure()
        p.plot(eh,hh)
        p.plot(ec[sb],h1[sb])
        p.plot(eh,eh*slope+intercept)
        p.loglog()
        p.ylim([h1.max()/50,h1.max()*2])
        p.xlim([10,1000])
        p.savefig("max2.png")
        
        plot.p.figure()
        plot.p.plot(ec,h1)
        
        s_le_peak=np.logical_and(ec>(emax1-10),ec<(emax1+10))
        plot.p.plot(ec[s_le_peak],h1[s_le_peak],lw=2)

        plot.p.plot(eh,hh)
        plot.p.plot(ec[sb],h1[sb])
        plot.p.plot(eh,eh*slope+intercept)
        plot.p.loglog()
        
        self.gain=(emax2-emax1)/(line2_energy-line1_energy) #
        self.offset=emax1-line1_energy*self.gain              # 
        
        self.igain=(line2_energy-line1_energy)/(emax2-emax1) #
        self.ioffset=line1_energy-emax1*self.igain              # 

        p.ylim([h1.max()/50,h1.max()*2])
        p.xlim([20,2000])
        plot.p.title("%.5lg ch %.5lg ch, off %.5lg ch, gain %.5lg ch/keV"%(emax1,emax2,self.offset,self.gain))
        plot.plot("bkgspec.png")


        sumfn="summary.txt"
        open(sumfn,"w").write(str(dict(offset=self.offset,gain=self.gain,peak_positions=[emax1,emax2])))

        self.max1=emax1
        self.max2=emax2
        
        self.summary=DataFile(sumfn)

        print(self.offset,"chan",self.gain,"chan/keV","peaks at",emax1,emax2) 


def rebin(a, shape):
    sh = shape[0],a.shape[0]//shape[0],shape[1],a.shape[1]//shape[1]
    #print "new shape",sh,a.shape
    return a.reshape(sh).sum(-1).sum(1)

class DetectorConditions(da.DataAnalysis):
    input_name="orbit"

    def main(self):
        if self.input_name.handle == "orbit":
            self.bias=120
            self.temperature=0
            return

        if self.input_name.handle == "gc+20C-100V":
            self.bias=100
            self.temperature=20
            return
        
        if self.input_name.handle == "saclay":
            self.input_saclay=100
            self.temperature=20
            return

        raise Exception("unacceptable!")

class BiparModel(ddosa.DataAnalysis):
    bipar_model=None

    def get_version(self):
        return self.get_signature()+"."+self.version+".bm"+(self.bipar_model.get_version() if self.bipar_model is not None else "undefined-unnp.loaded-"+repr(id(None))) 

    def main(self):
        pass

import imp
class LocateBiparModel(ddosa.DataAnalysis):
    mod=""
    _da_settings=['mod']
    run_for_hashe=True


    #bipar_attributes={}
    bipar_attributes={'rtbump':False}

    def main(self):
        try:
            bm = imp.load_source('bipar_model', os.environ['EDDOSA_TOOLS_DIR']+'/lut2model/python/bipar_model.py') # hc!
            for n,v in list(self.bipar_attributes.items()):
                setattr(bm,n,v)
            return BiparModel(use_bipar_model=bm)
        except Exception as e:
            traceback.print_stack()
            print("\033[31munable to np.load bipar model!\033[0m")

class PrintBiparModel(ddosa.DataAnalysis):
    input_biparmodel=LocateBiparModel

    def main(self):
        print(self.input_biparmodel.get_version())

class LEComplexBias(ddosa.DataAnalysis):
    bias=1
    
    def is_noanalysis(self):
        if self.bias==0:
            return True
        return False

    _da_settings=['bias']

class Fit3DModel(ddosa.DataAnalysis):
    input_p=FindPeaks
    input_fit=Fit1DSpectrumRev # or bipar
    #input_bkgspec=BinBackgroundSpectrum
    input_histograms=Bipar
    input_bias=LEComplexBias

    watched_analysis=True

    input_biparmodel=LocateBiparModel
    input_detector_cond=DetectorConditions

    version="v14.3"
    #version="v14.1"

    estimate_energy_power=0
    
    estimation_h=True
        
    only_estimation=False
    save_corrected_est=False


    def get_version(self):
        version=self.get_signature()+"."+self.version
        if self.only_estimation:
            version+=".onlyest"
        if self.save_corrected_est:
            version+=".np.saveest"
        if self.estimate_energy_power!=0:
            version+=".estpower%.5lg"%self.estimate_energy_power
        return version

    #cache=cache_local
    cached=True

    debug=False

    def plot_data_smoothed(self):
# plot data and contours
        #bkgbip=np.load(getattr(self.input_bkgspec,'h2_ISGRI_PHA1_ISGRI_RT1_1_1_').path)
        bkgbip=self.data
        line_model=self.lines_model
        m=gaussian_filter(bkgbip,2)

        plot.p.figure()
        print("max:",m.max())
        lvls=np.linspace(0,np.log10(m.max()),100)
        print(lvls)
        plot.p.contourf(np.log10(m+1e-10).np.transpose(),levels=lvls)
        plot.p.contour(np.log10(line_model).np.transpose(),levels=np.linspace(0,np.log10(line_model.max()),10),color="green")
        plot.p.xlim([60,1200])
        plot.p.ylim([10,150])
        plot.plot("data_bip.png")
        plot.p.xlim([60,160])
        plot.plot("data_bip_le.png")

    def load_data(self):
        self.data_raw=self.input_histograms.get_h2_pha1_pi()
        unzoomfactor=(1,1)
        #reducedshape=(2048/unzoomfactor[0],256/unzoomfactor[1])

        pha_coord,rt_coord=np.mgrid[:2048,:256]

        self.data=self.data_raw
        #self.data=rebin(self.data_raw,reducedshape)
        self.pha_coord=pha_coord
        #self.pha_coord=rebin(pha_coord,reducedshape)/unzoomfactor[1]/unzoomfactor[0]
        self.rt_coord=rt_coord
        #self.rt_coord=rebin(rt_coord,reducedshape)/unzoomfactor[1]/unzoomfactor[0]

    def assume_detector_conditions(self):
        self.detector_guess1.V=self.input_detector_cond.bias
        #self.input_detector_cond.temperature

    def estimate_lines(self):
        data=self.data

        print("data:",data)

        pha_coord=self.pha_coord
        rt_coord=self.rt_coord
        
        # search for he peak
        data_g1=gaussian_filter(data,(3,1))
        data_g5=gaussian_filter(data,(10,2))

        data_e20=ndimage.grey_erosion(data_g5, size=(20,5))
        data_e20_g5=gaussian_filter(data_e20,(20,4))
        data_d20=ndimage.grey_dilation(data_g5, size=(20,5))

        det=(data_g5-data_e20_g5)/data_g5

        lrt_peak_he=self.input_p.max2 

        he_min_pha=lrt_peak_he*0.9
        he_max_pha=lrt_peak_he*1.3
        he_min_rt=20
        he_max_rt=120

        print("he_max_pha",he_max_pha)

        he_slope_guess=4

        det_he=copy(det)
        #det_he[0:he_min_pha]=0
        det_he[int(he_max_pha):]=0
        det_he[pha_coord<he_min_pha-rt_coord*he_slope_guess]=0
        
        he_line_profile=[]

        for rt_i in np.arange(rt_coord.shape[1]):
            if rt_coord[0,rt_i]<he_min_rt: continue
            if rt_coord[0,rt_i]>he_max_rt: break
            peak_pha_i=det_he[:,rt_i].argmax()

            if peak_pha_i<10: continue

            # limit pha for early rt
            if rt_i<50:
                if abs(peak_pha_i+(rt_i-40)*he_slope_guess-lrt_peak_he)>lrt_peak_he*0.05:
                    peak_pha_i=lrt_peak_he
                

            print(rt_i,peak_pha_i)
            he_line_profile.append([rt_i,peak_pha_i,det_he[int(peak_pha_i),int(rt_i)]])

            
        
        # search for le peak
        le_min_pha_early_guess=70
        le_max_pha_early_guess=140

        range_reduction_factor=self.input_p.max1/(60.*2.)

        le_min_pha=le_min_pha_early_guess*range_reduction_factor
        le_max_pha=le_max_pha_early_guess*range_reduction_factor

        print(("will search for LE line in PHA range",le_min_pha,le_max_pha,"LE gain loss factor",range_reduction_factor))

        le_min_rt=20
        le_max_rt=105

        det_le=copy(data)
        det_le[0:int(le_min_pha)]=0
        det_le[int(le_max_pha):]=0
        
        le_line_profile=[]
        for rt_i in np.arange(rt_coord.shape[1]):
            if rt_coord[0,rt_i]<le_min_rt: continue
            if rt_coord[0,rt_i]>le_max_rt: break
            peak_pha_i=det_le[:,rt_i].argmax()

            print(rt_i,peak_pha_i)
            le_line_profile.append([rt_i,peak_pha_i,det_le[peak_pha_i,rt_i]])

        
        self.he_line_profile_estimate=array(he_line_profile)
        self.le_line_profile_estimate=array(le_line_profile)
        self.save_region_file_profiles("estimate")

        
        pyfits.HDUList([
                            pyfits.PrimaryHDU(data),
                            pyfits.ImageHDU(data_g5),
                            pyfits.ImageHDU(det),
                            pyfits.ImageHDU(det_he)
                    ]).writeto("smoothed_data.fits",overwrite=True)

        

    def main(self):
        self.le_complex_bias=self.input_bias.bias
# np.load
        self.load_data()

        if self.estimation_h:
            self.estimate_lines()
            
            self.correct_data_linear_estimation_update(tag="")
            self.correct_data_linear_estimation_update(tag="_p1")
            self.correct_data_linear_estimation_update(tag="_final")

           # self.correct_data_from_estimation(tag="")
           # self.correct_data_from_estimation(tag="_p1")
           # self.correct_data_from_estimation(tag="_final")

        if self.only_estimation:
            return
        
# estimate detector from peaks
        if hasattr(self.input_p,'max1'):
            self.detector_guess1=self.input_biparmodel.bipar_model.estimate_parameters_from_peaks(self.input_p.max1,self.input_p.max2) 
        else:
            self.detector_guess1=self.input_biparmodel.bipar_model.detector() 
        self.detector=self.detector_guess1 

        self.assume_detector_conditions()
        

        print("first guess:",self.detector_guess1)


        #self.plot_models(20,40,30)
        #self.plot_models(55,65,5)

#        self.plot_data_smoothed()



                
       # for scaling in np.linspace(0.8,1.2,10):
        #    residual_func([1,1,1,scaling])
   #         plot_model(det,1,compute_model(det,1,limrt=50))


        self.minrt=16
        self.maxrt=80

        self.parameters=[
                ['d_mu_e',[True,1.,0.8,1.2,
                        lambda self,x:setattr(self.detector,'mu_e',self.detector_guess1.mu_e*x),
                        lambda self:self.detector.mu_e/self.detector_guess1.mu_e]],
                ['d_mu_t',[True,2.,1.5,3,
                        lambda self,x:setattr(self.detector,'mu_t',self.detector_guess1.mu_t*x),
                        lambda self:self.detector.mu_t/self.detector_guess1.mu_t]],
                ['d_tau_e',[True,1.,0.5,1.5,
                        lambda self,x:setattr(self.detector,'tau_e',self.detector_guess1.tau_e*x),
                        lambda self:self.detector.tau_e/self.detector_guess1.tau_e]],
                ['d_tau_t',[True,1.,0.5,1.5,
                        lambda self,x:setattr(self.detector,'tau_t',self.detector_guess1.tau_t*x),
                        lambda self:self.detector.tau_t/self.detector_guess1.tau_t]],
                ['d_offset',[True,-2.5,-30,30,
                        lambda self,x:setattr(self.detector,'offset',self.detector_guess1.offset+x),
                        lambda self:(self.detector.offset-self.detector_guess1.offset)]]
         #       ['d_shape_0',[True,0,-1,1,
         #               lambda self,x:setattr(self.detector,'shape_0',self.detector_guess1.shape_0+x),
          #              lambda self:(self.detector.shape_0-self.detector_guess1.shape_0)]]
           ]

        self.detector=deepcopy(self.detector_guess1)
        self.set_free_pars([v[1][1] for v in self.list_free_pars()])
        
        self.minrt,self.maxrt=25,80
        self.plot_model(tag="first_guess")

        energies=[59.5,70,511]

        channels=self.input_biparmodel.bipar_model.get_chan(self.detector,energies,render_model="default")
        print(channels)

        print([self.input_fit.chan_to_energy(chan) for chan in channels])



        self.optimize_go=False
        #self.optimize()
        #self.optimize_go=False
        self.optimize_shape=True
        self.optimize() #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        self.he_line_file=da.DataFile("he_line_profiles.txt")
        self.le_line_file=da.DataFile("le_line_profiles.txt")

        self.minrt,self.maxrt=20,50
       # self.optimize()
        
        
        self.minrt,self.maxrt=10,200
        self.correct_data_linear_frommodel()

        self.minrt,self.maxrt=30,80
        self.plot_model()
        self.minrt,self.maxrt=50,80
        self.plot_model(tag="_hirt")
        self.minrt,self.maxrt=30,50
        self.plot_model(tag="_lort")


        self.np.explicit_output=['detector','fit_residuals','he_line_profile_estimate','le_line_profile_estimate','model_he_line_profile','model_le_line_profile','he_line_file','le_line_file']
            
    def correct_data_linear_estimation_update(self,tag):
        rt_prof_le=self.le_line_profile_estimate[:,0]
        rt_prof_he=self.he_line_profile_estimate[:,0]
        pha_prof_le=self.le_line_profile_estimate[:,1]
        pha_prof_he=self.he_line_profile_estimate[:,1]
        (rt_prof_le,pha_prof_le),(rt_prof_he,pha_prof_he)=self.correct_data_linear(((rt_prof_le,pha_prof_le),(rt_prof_he,pha_prof_he)),tag)
        self.le_line_profile_estimate[:,1]=array(pha_prof_le)
        self.he_line_profile_estimate[:,1]=array(pha_prof_he)
        self.save_region_file_profiles("_estimation_"+tag,color="red")

   
    def correct_data_linear_frommodel(self):
        rt_prof_le=list(zip(*self.model_le_line_profile))[0]
        pha_prof_le=list(zip(*self.model_le_line_profile))[5]
        rt_prof_he=list(zip(*self.model_he_line_profile))[0]
        pha_prof_he=list(zip(*self.model_he_line_profile))[5]

        (rt_prof_le,pha_prof_le),(rt_prof_he,pha_prof_he)=self.correct_data_linear(((rt_prof_le,pha_prof_le),(rt_prof_he,pha_prof_he)),"frommodel")

        #self.le_line_profile_estimate[:,1]=array(pha_prof_le)
        #self.he_line_profile_estimate[:,1]=array(pha_prof_he)

    def correct_data_linear(self,line_models,tag):
        rt_coord=self.rt_coord
        pha_coord=self.pha_coord
        pha_1d=pha_coord.transpose()[0]
        data=self.data
        
        (rt_prof_le,pha_prof_le),(rt_prof_he,pha_prof_he)=line_models
        
        data_corrected=np.zeros_like(data)
        data_uncorrected=np.zeros_like(data)
        
        self.energy_le=59.0+self.le_complex_bias
        self.energy_he=511.0
        
        energies=np.linspace(0,1024,pha_1d.shape[0])
        energy_le=self.energy_le
        energy_he=self.energy_he

        interp_le_line_profile=UnivariateSpline(rt_prof_le,pha_prof_le,k=1)
        interp_he_line_profile=UnivariateSpline(rt_prof_he,pha_prof_he,k=1)
        
        for irt in range(256):
            pha_he=interp_he_line_profile(irt)
            pha_le=interp_le_line_profile(irt)

            energy_scaled=energy_le+(pha_1d-pha_le)*(energy_he-energy_le)/(pha_he-pha_le)

            spectrum=interp1d(energy_scaled,data[:,irt],bounds_error=False)(energies)
            data_uncorrected[:,irt]=copy(data[:,irt])
            spectrum[np.isnan(spectrum)]=0
            data_corrected[:,irt]=spectrum

        self.save_corrected(energies,data_corrected,data_uncorrected,tag)
        he_line_biases_i=self.evaluate_biases_corrected(energies,data_corrected,tag)
        
        new_pha_prof_he=[]

        for rt,pha in zip(rt_prof_he,pha_prof_he):
            c_pha=he_line_biases_i(rt)*2 # energy to pha!
            new_pha_prof_he.append(pha+c_pha)

        return (rt_prof_le,pha_prof_le),(rt_prof_he,new_pha_prof_he)

    def save_corrected(self,energies,data_corrected,data_uncorrected,tag):
        np.savetxt("data_corrected_1d.txt",np.column_stack((energies,data_corrected.sum(axis=1),data_uncorrected.sum(axis=1))))

        fn="data_corrected%s.fits"%tag
        pyfits.HDUList( [pyfits.PrimaryHDU(data_corrected),
                        pyfits.BinTableHDU.from_columns([
                                pyfits.Column(name='EBOUND', format='E', array=energies)
                        ])]).writeto(fn,overwrite=True)

        setattr(self,'data_corrected_'+tag,da.DataFile(fn))
        pyfits.PrimaryHDU(data_uncorrected).writeto("data_uncorrected%s.fits"%tag,overwrite=True)
        
        d1d=data_corrected.sum(axis=1)
        ogip.spec.PHAI(d1d,sqrt(d1d),1).write("data_corrected_1d%s.fits"%tag)
        

    def evaluate_biases_corrected(self,energies,data_corrected,tag):
        rtedges=[10,40,50,70,100]
        energy_he=self.energy_he
        energy_le=self.energy_le

        he_line_biases=[]
        
        for rt1,rt2 in zip(rtedges[:-1],rtedges[1:]):
            d1d=data_corrected[:,rt1:rt2].sum(axis=1)
            
            on_line=np.zeros_like(d1d).astype(bool)
            on_line[np.logical_and(energies>self.energy_he-50,energies<energy_he+50)]=True

            on_region=np.zeros_like(d1d).astype(bool)
            on_region[np.logical_and(energies>energy_he-120,energies<energy_he+200)]=True

            on_bkg=np.logical_and(on_region,~on_line)

            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(energies[on_bkg]),np.log(d1d[on_bkg])) # validate
            bkg=np.exp(intercept+np.log(energies)*slope)
        
            # should check also if the  background is good
        
            smoothing=20
            mean_energy=energies[on_region][g1d((d1d-bkg)[on_region]*energies[on_region]**self.estimate_energy_power,smoothing).argmax()] #
            
            print("mean energy:",rt1,rt2,mean_energy,mean_energy-energy_he)

            he_line_biases.append([(rt1+rt2)/2.,mean_energy-energy_he])# 

            np.savetxt("estimation_%i_%i_%s.txt"%(rt1,rt2,tag),np.column_stack((energies[on_region],bkg[on_region],d1d[on_region],g1d((d1d-bkg)[on_region],smoothing))))

        he_line_biases_i=UnivariateSpline(*list(zip(*he_line_biases)),k=1)
    
        return he_line_biases_i

                
  #      np.savetxt("biases_he.txt",biases_i)
        
 #       self.save_region_file_profiles("fit_bias_%s"%tag)

    def correct_data_from_estimation(self,tag=""):
        rt_coord=self.rt_coord
        pha_coord=self.pha_coord
        pha_1d=pha_coord.transpose()[0]
        data=self.data

        rt_prof_le=list(zip(*self.le_line_profile_estimate))[0]
        rt_prof_he=list(zip(*self.he_line_profile_estimate))[0]
        model_prof_le=list(zip(*self.le_line_profile_estimate))[1]
        model_prof_he=list(zip(*self.he_line_profile_estimate))[1]
        model_amp_prof_he=list(zip(*self.he_line_profile_estimate))[2]

        data_corrected=np.zeros_like(data)
        data_uncorrected=np.zeros_like(data)

        energies=np.linspace(0,1024,pha_1d.shape[0])
        energy_le=self.energy_le
        energy_he=self.energy_he

        interp_le_line_profile=UnivariateSpline(rt_prof_le,model_prof_le,k=1)
        interp_he_line_profile=UnivariateSpline(rt_prof_he,model_prof_he,k=1)

        # correct
        for irt in range(256):
            pha_he=interp_he_line_profile(irt)
            pha_le=interp_le_line_profile(irt)

            energy_scaled=energy_le+(pha_1d-pha_le)*(energy_he-energy_le)/(pha_he-pha_le)

            spectrum=interp1d(energy_scaled,data[:,irt],bounds_error=False)(energies)
            data_uncorrected[:,irt]=copy(data[:,irt])
            spectrum[np.isnan(spectrum)]=0
            data_corrected[:,irt]=spectrum

       # self.data_corrected=data_corrected

        np.savetxt("data_corrected_1d.txt",np.column_stack((energies,data_corrected.sum(axis=1),data_uncorrected.sum(axis=1))))

        fn="data_corrected%s.fits"%tag
        pyfits.HDUList( [pyfits.PrimaryHDU(data_corrected),
                        pyfits.BinTableHDU.from_columns([
                                pyfits.Column(name='EBOUND', format='E', array=energies)
                        ])]).writeto(fn,overwrite=True)

        if self.save_corrected_est:
            self.data_corrected=da.DataFile(fn)
        self.data_corrected_fromestimation=da.DataFile(fn)

        pyfits.PrimaryHDU(data_uncorrected).writeto("data_uncorrected%s.fits"%tag,overwrite=True)
        
        
        d1d=data_corrected.sum(axis=1)
        ogip.spec.PHAI(d1d,sqrt(d1d),1).write("data_corrected_1d%s.fits"%tag)
        
        rtedges=[10,40,50,70,100]

        he_line_biases=[]

        
        for rt1,rt2 in zip(rtedges[:-1],rtedges[1:]):
            d1d=data_corrected[:,rt1:rt2].sum(axis=1)
            
            on_line=np.zeros_like(d1d).astype(bool)
            on_line[np.logical_and(energies>energy_he-50,energies<energy_he+50)]=True

            on_region=np.zeros_like(d1d).astype(bool)
            on_region[np.logical_and(energies>energy_he-200,energies<energy_he+200)]=True

            on_bkg=np.logical_and(on_region,~on_line)

            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(energies[on_bkg]),np.log(d1d[on_bkg])) # validate
            bkg=np.exp(intercept+np.log(energies)*slope)
        
            # should check also if background is good
        
            smoothing=20
            mean_energy=energies[on_region][g1d((d1d-bkg)[on_region],smoothing).argmax()]
            
            print("mean energy:",rt1,rt2,mean_energy,mean_energy-energy_he)

           # he_line_biases.append([(rt1+rt2)/2.,5+((rt1+rt2)/2-20)/80*30]) ## noo!!
            he_line_biases.append([(rt1+rt2)/2.,mean_energy-energy_he])# 

            # get new np.average and variance
 #           energies[on_region],bkg[on_region],d1d[on_region]

            np.savetxt("estimation_%i_%i_%s.txt"%(rt1,rt2,tag),np.column_stack((energies[on_region],bkg[on_region],d1d[on_region])))

           # ogip.spec.PHA(d1d,sqrt(d1d),1).write("data_corrected_1d_%i_%i.fits"%(rt1,rt2))

        he_line_biases_i=UnivariateSpline(*list(zip(*he_line_biases)),k=1)

        new_model_prof_he=[]

        biases_i=[]

        for rt,pha,amp in zip(rt_prof_he,model_prof_he,model_amp_prof_he):
            c_pha=he_line_biases_i(rt)*2 # energy to pha!
            biases_i.append([rt,pha,c_pha])

            new_model_prof_he.append([rt,pha+c_pha,amp])

        self.he_line_profile_estimate=new_model_prof_he
                
        np.savetxt("biases.txt",biases_i)
        
        self.save_region_file_profiles("fit_bias_%s"%tag)


        he_line_profile_i=UnivariateSpline(rt_prof_he,list(zip(*self.he_line_profile_estimate))[1],k=1)
        le_line_profile_i=UnivariateSpline(rt_prof_le,list(zip(*self.le_line_profile_estimate))[1],k=1)

   #     print rt_prof_he
  #      print self.he_line_profile_estimate

        # estimate binned

        rtedges=[10,40,50,70,100]

        he_binned_line_profile=[]
        le_binned_line_profile=[]

        for rt1,rt2 in zip(rtedges[:-1],rtedges[1:]):
            d1d=data_corrected[:,rt1:rt2].sum(axis=1)

            av_peak_pha=np.average([he_line_profile_i(rt) for rt in range(rt1,rt2)])
            pha_1d=pha_coord[:,0]

 #           print "pha_1d",[he_line_profile_i(rt) for rt in range(rt1,rt2)]
            
            # adjust width? 
            on_line=np.zeros_like(d1d).astype(bool)
            on_line[(pha_1d>av_peak_pha-50) & (pha_1d<av_peak_pha+50)]=True

            on_region=np.zeros_like(d1d).astype(bool)
            on_region[(pha_1d>av_peak_pha-200) & (pha_1d<av_peak_pha+200)]=True

            on_bkg=(on_region & ~on_line)

            slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(pha_1d[on_bkg]),np.log(d1d[on_bkg])) # validate
            bkg=np.exp(intercept+np.log(pha_1d)*slope)
        
            # should check also if background is good
        
            smoothing=20
            mean_pha=pha_1d[on_region][g1d((d1d-bkg)[on_region],smoothing).argmax()]

  #          print "mean pha:",rt1,rt2,mean_pha,"was",av_peak_pha

           # he_line_biases.append([(rt1+rt2)/2.,5+((rt1+rt2)/2-20)/80*30]) ## noo!!
            #he_line_biases.append([(rt1+rt2)/2.,mean_energy-energy_he])#             
            he_binned_line_profile.append([rt1,rt2,mean_pha])

            np.savetxt("pha_estimation_%i_%i_%s.txt"%(rt1,rt2,tag),np.column_stack((pha_1d[on_region],bkg[on_region],d1d[on_region])))

           # ogip.spec.PHA(d1d,sqrt(d1d),1).write("data_corrected_1d_%i_%i.fits"%(rt1,rt2))
            
        np.savetxt("pha_estimation_%s.txt"%tag,array(he_binned_line_profile))

    cid=0
    colors=['green','red','blue','magenta','cyan','white']

    def get_next_color(self):
        self.cid+=1
        if self.cid>=len(self.colors): self.cid=0
        return self.colors[self.cid]

    def correct_data(self):
        self.compute_model()

        rt_coord=self.rt_coord
        pha_coord=self.pha_coord
        pha_1d=pha_coord.transpose()[0]
        data=self.data
        det=self.detector

        rt_prof_le=list(zip(*self.model_le_line_profile))[0]
        model_prof_le=list(zip(*self.model_le_line_profile))[5]
        rt_prof_he=list(zip(*self.model_he_line_profile))[0]
        model_prof_he=list(zip(*self.model_he_line_profile))[5]

        data_corrected=np.zeros_like(data)
        data_uncorrected=np.zeros_like(data)
        
        self.energy_le=59.0+self.le_complex_bias
        self.energy_he=511.0
        energy_he=self.energy_he
        energy_le=self.energy_le

        energies=np.linspace(10,1000,pha_1d.shape[0])

        for rt,pha_le,pha_he in zip(rt_prof_le,model_prof_le,model_prof_he):
        #for irt,(rt,pha_le,pha_he) in enumerate(zip(rt_prof_le,model_prof_le,model_prof_he)):
            #irt=rt
            irt=list(rt_coord[0,:]).index(rt)
            print(irt)


            print("data row",data[:,irt])

            print("scaling",pha_1d)
            print("such as",energy_le,pha_le,energy_he,pha_he)

            energy_scaled=energy_le+(pha_1d-pha_le)*(energy_he-energy_le)/(pha_he-pha_le)

            print("scaled energy:",energy_scaled)

            spectrum=interp1d(energy_scaled,data[:,irt],bounds_error=False)(energies)
            print("scaled spectum:",spectrum)
            data_uncorrected[:,irt]=copy(data[:,irt])
            data_uncorrected[pha_le,irt]=1000
            data_uncorrected[pha_he,irt]=2000
            data_corrected[:,irt]=spectrum


        np.savetxt("data_corrected_1d.txt",np.column_stack((energies,data_corrected.sum(axis=1),data_uncorrected.sum(axis=1))))

        pyfits.PrimaryHDU(data_uncorrected).writeto("data_uncorrected_frommodel.fits",overwrite=True)
        
        fn="data_corrected_frommodel.fits"
        pyfits.HDUList( [pyfits.PrimaryHDU(data_corrected),
                        pyfits.BinTableHDU.from_columns([
                                pyfits.Column(name='EBOUND', format='E', array=energies)
                        ])]).writeto(fn,overwrite=True)

        self.data_corrected=da.DataFile(fn)
        self.data_corrected_frommodel=da.DataFile(fn)
        
        d1d=data_uncorrected.sum(axis=1)
        ogip.spec.PHAI(d1d,sqrt(d1d),1).write("data_uncorrected_1d.fits")
        
    def residual_func(self,pars):
        print(render('{RED}trying factors{/}'), pars)
        if any([np.isnan(p) for p in pars]):

            self.plot_model()
            raise Exception("given nan pars! "+str(self.nattempts))

        self.set_free_pars(pars)
        print(render('{RED}trying detector{/}'), self.detector)


        if self.optimize_go:
            energies=[self.energy_le,self.energy_he]

            channels=self.input_biparmodel.bipar_model.get_chan(self.detector,energies,render_model="default")

            renergies=[self.input_fit.chan_to_energy(chan) for chan in channels]
            print(renergies)
            value=-sum([((renergy-energy)/energy)**2 for renergy,energy in zip(renergies,energies)])**0.5

            open('fit.txt', 'a').write('%i' % self.nattempts + ' ' + ' '.join([ '%.5lg' % p for p in [value] + list(pars) + list(renergies) ]) + ' ' + str([self.detector]) + '\n')
        elif self.optimize_shape:
            energies=[self.energy_le,self.energy_he]

            mg=self.compute_model(maxrt=self.maxrt, minrt=self.minrt)

            stat=0
            
            max_model_amp=max(list(zip(*self.model_he_line_profile))[2])
            max_data_amp=max(list(zip(*self.he_line_profile_estimate))[2])

            f=open("he_line_profiles.txt","w")


            he_max_difference=0
            he_mean_difference=0
            he_wmean_difference=0
            n=0
            nw=0
            for rt in range(256):
                x=[a for a in self.model_he_line_profile if a[0]==rt]
                if x==[]: continue
                he_pha_model=float(x[0][5])
                he_pha_model_amp=float(x[0][2])

                x=[a for a in self.he_line_profile_estimate if a[0]==rt]
                if x==[]: continue
                he_pha_data=float(x[0][1])
                he_pha_data_amp=float(x[0][2])

                print("line profile:",rt,he_pha_data,he_pha_model,he_pha_model_amp)
                stat+=(he_pha_data-he_pha_model)**2*he_pha_data_amp/max_data_amp
                f.write("%.5lg %.5lg %.5lg %.5lg\n"%(rt,he_pha_data,he_pha_model,he_pha_data_amp))
                
                # stats
                if abs(he_pha_data-he_pha_model)>he_max_difference:  he_max_difference=he_pha_data-he_pha_model
                he_mean_difference+=(he_pha_data-he_pha_model)**2
                he_wmean_difference+=(he_pha_data-he_pha_model)**2*he_pha_data_amp**2
                nw+=he_pha_data_amp
                n+=1
            if n>0:
                he_mean_difference=sqrt(he_mean_difference/n)
                he_wmean_difference=sqrt(he_wmean_difference/nw)


            f.close()

            max_model_amp=max(list(zip(*self.model_le_line_profile))[2])
            max_data_amp=max(list(zip(*self.le_line_profile_estimate))[2])
            
            f=open("le_line_profiles.txt","w")

            le_max_difference=0
            le_mean_difference=0
            le_wmean_difference=0
            n=0
            nw=0

            bias_weight_le=5

            for rt in range(256):
                x=[a for a in self.model_le_line_profile if a[0]==rt]
                if x==[]: continue
                le_pha_model=float(x[0][5])
                le_pha_model_amp=float(x[0][2])

                x=[a for a in self.le_line_profile_estimate if a[0]==rt]
                if x==[]: continue
                le_pha_data=float(x[0][1])
                le_pha_data_amp=float(x[0][2])

                print("line profile:",rt,le_pha_data,le_pha_model,le_pha_model_amp)
                stat+=(le_pha_data-le_pha_model)**2*(le_pha_data_amp/max_data_amp)*bias_weight_le
                f.write("%.5lg %.5lg %.5lg %.5lg\n"%(rt,le_pha_data,le_pha_model,le_pha_data_amp))
                
                # stats
                if abs(le_pha_data-le_pha_model)>le_max_difference:  le_max_difference=le_pha_data-le_pha_model
                le_mean_difference+=(le_pha_data-le_pha_model)**2
                le_wmean_difference+=(le_pha_data-le_pha_model)**2*le_pha_data_amp
                nw+=le_pha_data_amp
                n+=1
            f.close()

            self.save_region_file_profiles("fit",self.model_he_line_profile,self.model_le_line_profile,positions=(0,5),color="blue")
            self.save_region_file_profiles("fit_data",color="red")

            if n>0:
                le_mean_difference=sqrt(le_mean_difference/n)
            if nw>0:
                le_wmean_difference=sqrt(le_wmean_difference/nw)
            
            self.fit_residuals=dict(
                                    he_mean_difference=he_mean_difference,he_max_difference=he_max_difference,he_wmean_difference=he_wmean_difference,
                                    le_mean_difference=le_mean_difference,le_max_difference=le_max_difference,le_wmean_difference=le_wmean_difference,
                )

            open("fit_residuals.txt","w").write(pprint.pformat(self.fit_residuals))

            value=-stat

            print(-value)

           # channels=bipar_model.get_chan(self.detector,energies,render_model="default")

           # renergies=[self.input_fit.chan_to_energy(chan) for chan in channels]
           # print renergies
            #value=-sum([((renergy-energy)/energy)**2 for renergy,energy in zip(renergies,energies)])**0.5

            open('fit.txt', 'a').write('%i' % self.nattempts + ' ' + ' '.join([ '%.5lg' % p for p in [value] + list(pars) ]) + ' ' + str([self.detector]) + '\n')
        else:
            mg=self.compute_model(maxrt=self.maxrt, minrt=self.minrt)
            sums = [mg[2].sum(), mg[1].sum(), mg[0].sum()]
            norms = mg[-1]
            open('fit.txt', 'a').write('%i' % self.nattempts + ' ' + ' '.join([ '%.5lg' % p for p in list(pars) + list(sums) + list(norms) ]) + ' ' + str([self.detector]) + '\n')
            value=(norms[0]/norms[4]+norms[1]/norms[5]) #*(mg[3])**0.5

        if self.nattempts % 50 == 0:
            self.plot_model(tag='_%.2i' % self.nattempts)
        
        if self.nattempts % 5 == 0:
            self.save_model()

        self.nattempts += 1

        self.set_free_pars(pars)

        print(render('{RED}returning{/}'), value)
        return value
    
    
    def optimize(self):
        open("fit.txt","w").close()

        self.nattempts=0 

        free_pars=list(zip(*self.list_free_pars()))[1]

        print("free parameters:")
        for k,v in self.list_free_pars():
            print(k,":",v[:4])

        opt = nlopt.opt(nlopt.LN_COBYLA, len(free_pars))
        lower_bounds=[v[2] for v in free_pars]
        upper_bounds=[v[3] for v in free_pars]
        print("lower bounds:",lower_bounds)
        print("upper bounds:",upper_bounds)

        opt.set_lower_bounds(lower_bounds)
        opt.set_upper_bounds(upper_bounds)
        opt.set_max_objective(lambda p,g:self.residual_func(p))
        opt.set_xtol_rel(1e-4)
    
       # x0=[v[1] for v in free_pars]
        x0=[v[5](self) for v in free_pars]
        print("x0",x0)
        x = opt.optimize(x0)
        optf = opt.last_optimum_value()
        print("optimum at ", x)
        print("optimum value = ", optf)
        print("result code = ", opt.last_optimize_result())
       
        self.set_free_pars(x)

        print("detector now:",self.detector)

    def set_free_pars(self,x):
        for i,v in enumerate(self.list_free_pars()):
            v[1][4](self,x[i])

    def get_free_pars(self):
        return [v[1][5](self) for v in self.list_free_pars()]

    def list_free_pars(self):
        return [[p,k] for p,k in self.parameters if k[0]]

    modeldict={}

    def monoenergetic_model(self,energy,resolutionfactor=1.):
        rt_coord=self.rt_coord
        pha_coord=self.pha_coord
        data=self.data
        det=self.detector

        detdef=repr(self.detector)
        if energy in self.modeldict:
            if self.modeldict[energy][0]==detdef:
                return self.modeldict[energy][1]

        print("for the line",energy)
        line_model=self.input_biparmodel.bipar_model.make_bipar_monoenergetic(det,energy,resolution_step=1,resolutionfactor=resolutionfactor,render_model="m0")
        line_model=rebin(line_model,self.data.shape)
        self.modeldict[energy]=[detdef,line_model]
        return line_model

    def plot_models(self,e1,e2,ne):
        #if (e1-ep)/e1<step: continue
        #if e1>maxen: break
        plot.p.clf()
        for energy in np.logspace(np.log10(e1),np.log10(e2),ne):
            line_model=self.monoenergetic_model(energy)
            pha=self.pha_coord[:,0]
            rt=self.rt_coord[0,:]
            apha=(line_model*self.pha_coord).sum(axis=0)/(line_model).sum(axis=0)
            amp=sum(line_model,axis=0)
            
            #print pha,rt,apha,amp

            plot.plot_line_colored(apha,rt,amp,cmap='autumn')
            #plot_module.plot_line_colored(what[ie][:,0],np.arange(256),ie*np.ones_like(what[ie][:,1]),1+3*what[ie][:,1]/np.average(what[:,0,1]))
            #p.xlim([0,maxen*2.5])
            plot.p.xlim([apha[~np.isnan(apha)].min(),apha[~np.isnan(apha)].max()])
            plot.p.ylim([0,256])
        plot.plot("edge_range_%.5lg_%.5lg.png"%(e1,e2))

        #p.plot(what[ie][:,0],np.arange(256))
        #p.scatter(what[ie][:,0],np.arange(256),c=what[ie][:,1])

    def mask_model(self,line_model,bkg_line_model=None):
        if bkg_line_model is None:
            bkg_line_model=line_model

        rt_coord=self.rt_coord
        pha_coord=self.pha_coord
        data=self.data
        det=self.detector

        mask_limits=np.logical_and(rt_coord<self.maxrt,rt_coord>self.minrt)
        
        mask=line_model>line_model.max()/10.
        mask=np.logical_and(mask,mask_limits)

        print("fraction in the line",np.where(mask)[0].shape[0]/1./line_model.flatten().shape[0])

        mask_bkg=np.logical_and(bkg_line_model>bkg_line_model.max()/10000,~mask)
        mask_bkg=np.logical_and(mask_bkg,mask_limits)
        
        print("fraction in the background",np.where(mask_bkg)[0].shape[0]/1./bkg_line_model.flatten().shape[0])

        return line_model,mask,mask_bkg
    
    def model_gains(self):
        energies=[22,30,57.9817,  59.3182,  67.2443,  72.8042,   74.9694,  84.9360, 150, 511]

        m0=self.monoenergetic_model(energies[0],resolutionfactor=1.)

        def gain(e1,e2,rt1=16,rt2=30):
            m1=self.monoenergetic_model(e1,resolutionfactor=1.)
            m2=self.monoenergetic_model(e2,resolutionfactor=1.)

            avch=lambda m:(sum((m*self.pha_coord)[:,rt1:rt2])/sum(m[:,rt1:rt2]))
            g=(avch(m1)-avch(m2))/(e1-e2)

            print(rt1,rt2,"gain",e1,e2,g)

        for rt1,rt2 in [(16,30),(30,50),(50,70),(70,116)]:
            gain(energies[0],energies[1],rt1,rt2)
            gain(energies[1],energies[2],rt1,rt2)
            gain(energies[0],energies[2],rt1,rt2)
            gain(energies[2],energies[3],rt1,rt2)
            gain(energies[2],energies[7],rt1,rt2)
            gain(energies[2],energies[8],rt1,rt2)
            gain(energies[2],energies[9],rt1,rt2)

    def le_model(self,legain=True):
        energies=[57.9817,  59.3182,  67.2443,  72.8042,   74.9694,  84.9360]

        for i in range(len(energies)):
            energies[i]+=self.le_complex_bias

        fractions_0_1=[0.365,0.635]

        e0=self.monoenergetic_model(energies[0],resolutionfactor=1.)
        e1=self.monoenergetic_model(energies[1],resolutionfactor=1.)
        le_model=e0*fractions_0_1[0]+e1*fractions_0_1[1]

        
        #e0_bkg=self.monoenergetic_model(energies[0],resolutionfactor=0.5)
        #e1_bkg=self.monoenergetic_model(energies[1],resolutionfactor=0.5)
        #le_model_bkg=e0*fractions_0_1[0]+e1*fractions_0_1[1]

        #elle=self.monoenergetic_model(20,resolutionfactor=10.)
        #pyfits.PrimaryHDU(elle).writeto("energy_lle.fits",overwrite=True)

        #raise

        return self.mask_model(le_model)
    
    def he_model(self):
        return self.mask_model(self.monoenergetic_model(511))


    def compute_model(self,maxrt=100,minrt=10,limpha=2048,debug=False):
        print("will compute two-line model for",self.detector)
        rt_coord=self.rt_coord
        pha_coord=self.pha_coord
        data=self.data
        det=self.detector

        line_model_1,mask_1,mask_bkg_1=self.le_model()
        line_model_2,mask_2,mask_bkg_2=self.he_model()

        self.model_gains()

        mask=np.logical_or(mask_1,mask_2)
        mask_bkg=np.logical_or(mask_bkg_1,mask_bkg_2)


        if False:
            plot.p.figure()
            plot.p.contourf(pha_coord.transpose(),rt_coord.transpose(),np.log10(mask_bkg).np.transpose(),levels=np.linspace(0,2,2),colors='r',alpha=0.3)
            #plot.p.contourf(pha_coord.transpose(),rt_coord.transpose(),np.log10(mask).np.transpose(),levels=np.linspace(0,2,2),colors='b',alpha=0.3)
            plot.p.xlim([60,1200])
            plot.p.ylim([10,150])
            plot.plot("mask.png")
            plot.p.xlim([60,160])
            plot.plot("mask_le.png")
        
        def normalize_model(model,data,mask,mask_bkg):
            d_bkg=copy(data)
            data=copy(data)
            bkg_model=np.zeros_like(d_bkg)
            mask_window=np.logical_or(mask,mask_bkg)
            data[~mask_window]=0

            model_normalized=copy(model)
            model_normalized[~mask]=0

            line_rt_profile=[]
            for i in range(mask_bkg.shape[1]):
                s=d_bkg[:,i][mask_bkg[:,i]]
                pha_s=pha_coord[:,i][mask_bkg[:,i]]
                pha_se=pha_coord[:,i][mask_window[:,i]]
                #av=np.average(s)
             #   print s
               # bkg_model[:,i]=av
             #   print "at",i,av,bkg_model[:,i]

                if pha_s.shape[0]>2:
                    try:
                        bkg_model[:,i][mask_window[:,i]]=interp1d(pha_s,s)(pha_se)
                    except ValueError:
                        bkg_model[:,i][mask_window[:,i]]=np.average(s)
                if pha_s.shape[0]==1:
                    bkg_model[:,i][mask_window[:,i]]=s[0]

                if pha_s.shape[0]==0:
                 #   print "no usable data at",i
                    continue 

                data_max=(d_bkg[:,i]-bkg_model[:,i])[mask_window[:,i]].max()
                data_phamax=pha_se[(d_bkg[:,i]-bkg_model[:,i])[mask_window[:,i]].argmax()]

                model_max_rt=model[:,i][mask_window[:,i]].max()

                model_avpha_rt=sum(model[:,i][mask_window[:,i]]*pha_se)/sum(model[:,i][mask_window[:,i]])

                model_phamax_rt=pha_se[model[:,i][mask_window[:,i]].argmax()]

                #model_normalized[:,i]=model[:,i]/model_max_rt*data_max


                line_rt_profile.append([rt_coord[:,i][0],data_max,model_max_rt,bkg_model[:,i].mean(),data_phamax,model_phamax_rt,model_avpha_rt])
                #print "model at rt",line_rt_profile[-1],model_normalized[:,i][mask_window[:,i]]

            print("total in bkg model",bkg_model.sum(),d_bkg[~np.isnan(d_bkg)].sum())
            
            # now normalize the rest
            d=copy(data)-copy(bkg_model)
            d[~mask_window]=0
            dn=d.sum(axis=1).max()
            
            model_masked=copy(model)
            model_masked[~mask_window]=0
            mn=model_masked.sum(axis=1).max()
            #mn=model_normalized.sum(axis=1).max()
            model_normalized=model/mn*dn


            bkg_model_masked=copy(bkg_model)
            bkg_model_masked[~mask]=0
            avbkg=bkg_model_masked.sum(axis=1).mean()

            if self.debug: 
                m=model_normalized
    
                print("total in bkg model",m.sum(),m.max())

                plot.p.figure()
                plot.p.subplot(211)
                plot.p.contourf(pha_coord.transpose(),rt_coord.transpose(),m.transpose(),levels=np.linspace(m.min(),m.max(),100))
                plot.p.xlim([60,1200])
                plot.p.ylim([10,150])
                plot.p.colorbar()
                plot.p.subplot(212)
                m_1d=m.sum(axis=1)
                pha_1d=pha_coord.transpose()[0]
                data_1d=data.sum(axis=1)
                bkgm_1d=bkg_model.sum(axis=1)
                model_1d=model.sum(axis=1)
                plot.p.plot(pha_1d,data_1d,label="data")
                plot.p.plot(pha_1d,m_1d+bkgm_1d,label="bkg model")
                plot.p.plot(pha_1d,model_1d,label="model")
                #plot.p.plot(pha_1d,model_1d+m_1d,label="model+bkgmodel")
                plot.p.xlim([30,200])
                plot.p.ylim([data_1d.max()/10.,data_1d.max()*1.5])
                #plot.p.xlim([80,1200])
         #       plot.p.legend(loc=3)
                plot.p.semilogy()   
                plot.plot("bkg.png")

                np.savetxt("model.txt",np.column_stack([pha_1d,data_1d,bkgm_1d,model_1d,m_1d]))
        
                np.savetxt("line_profile.txt",array(line_rt_profile),delimiter=" ")

 #               raise

#                   plot.p.xlim([80,160])
#                  plot.plot("bkg_le.png")


            print("normalized, bkg:",model_normalized.sum(),bkg_model.sum())

            return model_normalized,bkg_model,dn/mn,dn,avbkg,line_rt_profile
            #return model/mn*dn,bkg_model,dn/mn,dn,line_rt_profile
               
        normalized_model_2,bkg_model_2,norm_2,datanorm_2,avbkg_2,line_rt_profile_2=normalize_model(line_model_2,data,mask_2,mask_bkg_2)
        normalized_model_1,bkg_model_1,norm_1,datanorm_1,avbkg_1,line_rt_profile_1=normalize_model(line_model_1,data,mask_1,mask_bkg_1)

        np.savetxt("line_profile_1.txt",array(line_rt_profile_1),delimiter=" ")
        np.savetxt("line_profile_2.txt",array(line_rt_profile_2),delimiter=" ")

        self.model_le_line_profile=line_rt_profile_1
        self.model_he_line_profile=line_rt_profile_2

        self.save_region_file_profiles()


        if False: 
            model=normalized_model_1
            bkg_model=bkg_model_1
            mask=mask_1
            mask_bkg=mask_bkg_1

            m=copy(bkg_model)
            m[~np.logical_or(mask,mask_bkg)]=0
            m[np.isnan(m)]=0

            print("total in bkg model",m.sum(),m.max())

            plot.p.figure()
            plot.p.subplot(211)
            plot.p.contourf(pha_coord.transpose(),rt_coord.transpose(),m.transpose(),levels=np.linspace(m.min(),m.max(),100))
            plot.p.xlim([80,1200])
            plot.p.ylim([10,150])
            plot.p.colorbar()
            plot.p.subplot(212)
            m_1d=m.transpose().sum(axis=0)
            pha_1d=pha_coord.transpose()[0]
            data_1d=data.mean(axis=1).np.transpose()
            model_1d=model.mean(axis=1).np.transpose()
            plot.p.plot(pha_1d,m_1d,label="bkg model")
            plot.p.plot(pha_1d,data_1d,label="data")
            #plot.p.plot(pha_1d,model_1d,label="model")
            #plot.p.plot(pha_1d,model_1d+m_1d,label="model+bkgmodel")
            plot.p.plot(pha_1d,model_1d+m_1d,label="model+bkgmodel")
            plot.p.xlim([30,200])
            plot.p.ylim([data_1d.max()/10.,data_1d.max()*1.5])
            #plot.p.xlim([80,1200])
     #       plot.p.legend(loc=3)
            plot.p.semilogy()   
            plot.plot("bkg.png")

            raise


        prob=0

        def comparison(model,data):
            return (model*data).sum() / (data**2).sum()**0.5 / (model**2).sum()**0.5

        prob+=comparison(normalized_model_1[mask_1]+bkg_model_1[mask_1],data[mask_1])

        print("prob:",prob)

        
        prob+=(normalized_model_2[mask_2]*data[mask_2]).sum() / (data[mask_2]**2).sum() / (normalized_model_2[mask_2]**2).sum()
        
        print("prob full:",prob)
        print("norms:",[norm_1,norm_2,datanorm_1,datanorm_2])

        model=np.zeros_like(data)
        model[mask_1]+=normalized_model_1[mask_1]
        model[mask_2]+=normalized_model_2[mask_2]
        
        bkg_model=np.zeros_like(data)

        mask_bkgline_1=np.logical_or(mask_bkg_1,mask_1)
        mask_bkgline_2=np.logical_or(mask_bkg_2,mask_2)
        bkg_model[mask_bkgline_1]+=bkg_model_1[mask_bkgline_1]
        bkg_model[mask_bkgline_2]+=bkg_model_2[mask_bkgline_2]
        
        if np.isnan(prob):
            prob=1e20

        
        return model,bkg_model,data,prob,mask,mask_bkg,[norm_1,norm_2,datanorm_1,datanorm_2,avbkg_1,avbkg_2]
        
    def save_region_file_profiles(self,tag="",he=None,le=None,positions=(0,1),color="green"):
        if he is None:
            he=self.he_line_profile_estimate
        if le is None:
            le=self.le_line_profile_estimate

        region_file_name="line_%s.reg"%tag
        region_file=open(region_file_name,"w")
        region_file.write("""
# Region file format: DS9 version 4.1
# Filename: smoothed_data.fits
global color=%s dashlist=8 3 width=1 font="helvetica 10 normal roman" select=1 highlite=1 dash=0 fixed=0 edit=1 move=1 delete=1 include=1 source=1
physical
"""%color)
        a=positions[0]
        b=positions[1]
            
        for i in range(1,len(he)):
            region_file.write("line(%.5lg,%.5lg,%.5lg,%.5lg) # line=0 0\n"%(he[i-1][a],he[i-1][b],he[i][a],he[i][b]))

        for i in range(1,len(le)):
            region_file.write("line(%.5lg,%.5lg,%.5lg,%.5lg) # line=0 0\n"%(le[i-1][a],le[i-1][b],le[i][a],le[i][b]))

        region_file.close()

    def save_model(self,tag="",pars=""):
        det=self.detector
        rt_coord=self.rt_coord
        pha_coord=self.pha_coord
        data=self.data

        pars=self.get_free_pars()
        
        model,bkg_model,data,prob,mask,mask_bkg,norms=self.compute_model()
        lvls=np.linspace(0,10,100)

        mask_outline=np.zeros_like(mask)
        mask_outline[mask]=1

        pyfits.HDUList([    pyfits.PrimaryHDU(model.np.transpose()),
                            pyfits.ImageHDU(data.np.transpose())]).writeto("model_%s.fits"%tag,overwrite=True)

        print("mask open",np.where(mask)[0].shape[0])

        #print pha_coord.shape,rt_coord.shape,m.transpose().shape
        m=copy(data)
        m[~np.logical_or(mask,mask_bkg)]=0

        tag="_rt%.5lg-%.5lg"%(self.minrt,self.maxrt)
            
        data_1d=m.sum(axis=1)
        model_1d=model.sum(axis=1)
        pha_1d=pha_coord.transpose()[0]
        bkg_model_1d=bkg_model.sum(axis=1)
        np.savetxt("projections_pha_"+tag+".txt",np.column_stack([pha_1d,data_1d,model_1d,bkg_model_1d]))

        data_1d=m.sum(axis=0)
        model_1d=model.sum(axis=0)
        rt_1d=rt_coord.transpose()[:,0]
        bkg_model_1d=bkg_model.sum(axis=0)
        np.savetxt("projections_rt_"+tag+".txt",np.column_stack([rt_1d,data_1d,model_1d,bkg_model_1d]))
        
    def plot_model(self,tag="",pars=""):
        self.save_model(tag=tag)

        det=self.detector
        rt_coord=self.rt_coord
        pha_coord=self.pha_coord
        data=self.data

        pars=self.get_free_pars()

        model,bkg_model,data,prob,mask,mask_bkg,norms=self.compute_model()
        lvls=np.linspace(0,10,100)
        print(lvls)

        mask_outline=np.zeros_like(mask)
        mask_outline[mask]=1

        print("mask open",np.where(mask)[0].shape[0])

        #print pha_coord.shape,rt_coord.shape,m.transpose().shape
        m=copy(data)
        m[~np.logical_or(mask,mask_bkg)]=0
    
        for xrange,rtag in  [([60,1200],""),([60,160],"_le"), ([600,1200],"_he")]:
            plot.p.figure()
            range_selection=np.logical_and(pha_coord>xrange[0],pha_coord<xrange[1])
            data_max=m[range_selection].max()
            data_av=np.average(m[np.logical_and(m>0,range_selection)])
            print("max:",data_max)
            print("np.average:",data_av)
            lvls=np.linspace(np.log10(data_av/10),np.log10(data_max),30)
            print(lvls)

            # 2d
            f,axes=plot.p.subplots(2,2)
            #plot.p.subplot(222)

            p=axes[0,0]
            p.contourf(pha_coord.transpose(),rt_coord.transpose(),np.log10(m+1e-10).np.transpose(),levels=lvls)
            p.contour(pha_coord.transpose(),rt_coord.transpose(),np.log10(model).np.transpose(),levels=np.linspace(0,np.log10(model.max()),10),cmap=plot.p.cm.PuBu)
            p.set_xlim(xrange)
            p.set_ylim([10,150])
            p.set_title(":%.5lg %s\ndet: %s\n%s"%(prob,norms,str(det),str(pars)),fontsize=10)
            
            # 2d
            #plot.p.subplot(222)

            p=axes[1,1]

            #for prof in (self.model_le_line_profile,self.model_he_line_profile,self.le_line_profile_estimate,self.he_line_profile_estimate):

            def plot_profile(profile,color,offset=0):
                rt=list(zip(*profile))[0]
                pha=list(zip(*profile))[1+offset]
                #datamax_prof=zip(*prof)[1]
                #modelmax_prof=zip(*prof)[2]
                p.plot(pha,rt,color=color)

            plot_profile(self.model_le_line_profile,"red",3)
            plot_profile(self.model_le_line_profile,"green",4)
            plot_profile(self.model_he_line_profile,"red",3)
            plot_profile(self.model_he_line_profile,"green",4)
            plot_profile(self.le_line_profile_estimate,"blue")
            plot_profile(self.he_line_profile_estimate,"blue")

            p.set_xlim(xrange)
            p.set_ylim([10,150])

            # 1d: pha
            data_1d=m.sum(axis=1).np.transpose()
            model_1d=model.sum(axis=1).np.transpose()
            pha_1d=pha_coord.transpose()[0]
            bkg_model_1d=bkg_model.sum(axis=1).np.transpose()

            p=axes[1,0]
            p.plot(pha_1d,data_1d,label="data")
            p.plot(pha_1d,model_1d,label="line model")
            p.plot(pha_1d,bkg_model_1d,label="bkg model")
            p.plot(pha_1d,bkg_model_1d+model_1d,label="bkg+line model")
            #plot.p.legend()
            p.set_xlim(xrange)
            maxdata=data_1d[np.logical_and(pha_1d>xrange[0],pha_1d<xrange[1])].max()
            y1,y2=maxdata*0.1,maxdata*1.3
            print("limits",y1,y2,maxdata)
            p.set_ylim([y1,y2])
            p.loglog()
            p.semilogy()
            
            # 1d: rt

            data_rt1d=m.sum(axis=0)
            model_rt1d=model.sum(axis=0)
            rt_1d=rt_coord.transpose()[:,0]
            bkg_model_rt1d=bkg_model.sum(axis=0)
            
            p=axes[0,1]
            p.plot(data_rt1d,rt_1d)
            p.plot(model_rt1d,rt_1d)
            p.plot(bkg_model_rt1d,rt_1d)
            p.plot(model_rt1d+bkg_model_rt1d,rt_1d)
            p.set_ylim([10,150])

            plot.plot("data_bip%s%s.png"%(rtag,tag))
                
class GenerateLUT2(ddosa.DataAnalysis): pass


class PlotLines(ddosa.DataAnalysis):
    input_response=GenerateLUT2

    def main(self):
        fn=self.input_response.response_3d.path

        f=pyfits.open(fn)

        e1=f[1].data['ENERGY']

        for en in [22, 59, 511]:
            i=abs(e1-en).argmin()

            m=f[0].data[i,:,:]

            plot.p.figure()
            plot.p.contourf(np.log10(m).np.transpose(),levels=np.linspace(0,np.log10(m.max()),100))
            plot.p.title("%.5lg keV"%(e1[i]))
            plot.plot("response_single_%.5lg.png"%en)

class ibis_isgr_energy_scw_xx(DataAnalysis):
    cached=False

    input_scw=ScWData
    input_events=ibis_isgr_energy
    input_ecorrdata=GetEcorrCalDB
    input_lut2=GenerateLUT2

    version="v6_extras_scw1"

    def main(self):
        event_file=pyfits.open(self.input_events.output_events.path)


        event_data=event_file[1].data
        print(event_file,self.input_events.output_events.path,event_data)
        lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())[0].data

        print(lut2)
        print(lut2.shape)    

        pha1,rt1=event_data['ISGRI_PHA1'],event_data['ISGRI_PI']
        pha2=event_data['ISGRI_PHA2']
    
        pha1[pha1>=2048]=0

        print(pha1.max(),rt1.max(),pha2.max())

        newenergy=lut2[pha2.astype(int),rt1.astype(int)]
        #event_file[1].insert_column(name='')
        event_file[1].data['ISGRI_ENERGY']=newenergy

        print(newenergy.max())

        fn="isgri_events_scw.fits"
        event_file.writeto(fn,overwrite=True)
        self.output_events=DataFile(fn)

class LUT2FromFile(DataAnalysis):
    input_fn=None
    
    lut2_3d=False

    def main(self):
        pass

    def lut2_1d_to_3d(self):
        if self.lut2_3d:
            return self.input_fn.handle
        else:
            # copied from GLT
            print("converting from 1d lut2")
            lut2=pyfits.open(self.input_fn.handle)[0].data
            #from scipy.ndimage import gaussian_filter
            #lut2=gaussian_filter(lut2,1)
            fd=pyfits.open(os.environ['CURRENT_IC']+"/ic/ibis/mod/isgr_3dl2_mod_0001.fits")
            for i in range(500):
                fd[1].data[i]=lut2.np.transpose()[:,::2]*30.
            fd.writeto("lut2_3d.fits",overwrite=True)
            return "lut2_3d.fits"

class FinalizeLUT2(da.DataAnalysis):
    pass

class FinalizeLUT2P4(da.DataAnalysis):
    pass


class ibis_isgr_energy_scw(DataAnalysis):
    cached=False

    input_scw=ScWData
    input_ecorrdata=GetEcorrCalDB
    input_lut2=FinalizeLUT2
    #input_lut2=GenerateLUT2
    
    lut2_3d=False

    copy_cached_input=False

    #version="v6_extras_scw3_lut23d"
    version="v9_extras_scw3"

    #generate_lut2_3d
    def get_version(self):
        return self.get_signature()+"."+self.version+\
                    ("_lut23d" if self.lut2_3d else ".lut22d")

    def main(self):
        if self.lut2_3d:
            lut2_3d_fn=self.input_lut2.lut2_3d.get_path()
        else:
            lut2_3d_fn=self.input_lut2.lut2_1d_to_3d()

        remove_withtemplate("isgri_events_corrected_scw2.fits(ISGR-EVTS-COR.tpl)")

        construct_gnrl_scwg_grp(self.input_scw,[\
           # self.input_scw.scwpath+"/isgri_events.fits[ISGR-EVTS-ALL]", \
            self.input_scw.scwpath+"/isgri_events.fits[ISGR-EVTS-ALL]" if not hasattr(self,'input_eventfile') else self.input_eventfile.evts.get_path(), \
            self.input_scw.scwpath+"/ibis_hk.fits[IBIS-DPE.-CNV]" \
        ])

        bin=os.environ['COMMON_INTEGRAL_SOFTDIR']+"/spectral/ibis_isgr_energy/ibis_isgr_energy_pha2_optdrift/ibis_isgr_energy"
        ht=heatool(bin)
        ht['inGRP']="og.fits"
        ht['outCorEvts']="isgri_events_corrected_scw2.fits(ISGR-EVTS-COR.tpl)"
        ht['useGTI']="n"
        ht['randSeed']=500
        ht['riseDOL']=lut2_3d_fn
        ht['GODOL']=self.input_ecorrdata.godol
        ht['supGDOL']=self.input_ecorrdata.supgdol
        ht['supODOL']=self.input_ecorrdata.supodol
        ht['chatter']="4"
        ht['corGainDrift']="n"
        ht.run()


        self.output_events=DataFile("isgri_events_corrected_scw2.fits")

class ibis_isgr_evts_tag_scw(ddosa.ibis_isgr_evts_tag):
    cached=False
    input_events_corrected=ibis_isgr_energy_scw

class ibis_isgr_energy_scw_P4(ibis_isgr_energy_scw):
    input_lut2 = FinalizeLUT2P4

class ibis_isgr_evts_tag_scw_P4(ibis_isgr_evts_tag_scw):
    cached=False
    input_events_corrected=ibis_isgr_energy_scw_P4

class VerifyLines(ddosa.DataAnalysis):
    pass

class VerifyLinesP4(ddosa.DataAnalysis):
    pass

class ISGRIEventsScW(ddosa.ISGRIEvents):
    input_verifylines=VerifyLines
    input_evttag=ibis_isgr_evts_tag_scw_P4
    #input_evttag=ibis_isgr_evts_tag_scw

    cached=True
   # cached=False
    cache=cache_local
    
    read_caches=[cache_local.__class__]
    write_caches=[cache_local.__class__]

    version="v4"
    def main(self):
        self.events=self.input_evttag.output_events

class ISGRIEventsScWP4(da.DataAnalysis):
    pass

class BinBackgroundSpectrumP2(BinBackgroundSpectrum):
    input_events=ISGRIEventsScW
    tag="P2"
    #cached=False
    #input_fit=Fit3DModel

    plot_essentials=True

    def plot_more(self):
       # line1=np.load(self.input_fit.line1.path)
      #  line2=np.load(self.input_fit.line2.path)

       # plot.p.figure()
       # plot.p.plot(line1[:116].sum(axis=1))
       # plot.p.plot(line2[:116].sum(axis=1))
       # plot.plot("lines.png")
        pass

class BinBackgroundSpectrumExtraP2(BinBackgroundSpectrumP2):
    save_extra=True


class BinBackgroundSpectrumExtra(BinBackgroundSpectrum):
    save_extra=True


class root(da.DataAnalysis):
    input_lines=PlotLines
    input_ISGRIEventsScW=ISGRIEventsScW
   # input_2dresp=Generate2DResponse



# merged

class BinBackgroundList(da.DataAnalysis):
    input_scwlist=ddosa.RevScWList

    copy_cached_input=False
    allow_alias=True

    input_will_use_promise=BasicEventProcessingSummary

    def main(self):
        self.thelist=[]
        for s in self.input_scwlist.scwlistdata:
            a=BinBackgroundSpectrum(assume=s)
            print(a,a.assumptions)
            self.thelist.append(a)

class BasicP2EventProcessingSummary(DataAnalysis):
    run_for_hashe=True

    def main(self):
        mf=ISGRIEventsScW(assume=ScWData(input_scwid="any",use_abstract=True)) # arbitrary choice of scw, should be the same: assumption of course
        ahash=mf.process(output_required=False,run_if_haveto=False)[0]
       # print "one scw hash:",ahash
        #ahash=dataanalysis.hashe_replace_object(ahash,'AnyScW','None')
        print("generalized hash:",ahash)
        rh=dataanalysis.shhash(ahash)
        print("reduced hash",rh)
        return [dataanalysis.DataHandle('processing_definition:'+rh[:8])]

class BasicP3EventProcessingSummary(DataAnalysis):
    run_for_hashe=True

    def main(self):
        mf=FineEnergyCorrection(assume=ScWData(input_scwid="any",use_abstract=True)) # arbitrary choice of scw, should be the same: assumption of course
        ahash=mf.process(output_required=False,run_if_haveto=False)[0]
       # print "one scw hash:",ahash
        #ahash=dataanalysis.hashe_replace_object(ahash,'AnyScW','None')
        print("generalized hash:",ahash)
        rh=dataanalysis.shhash(ahash)
        print("reduced hash",rh)
        return [dataanalysis.DataHandle('processing_definition:'+rh[:8])]

class BinBackgroundListP2(da.DataAnalysis):
    input_scwlist=ddosa.RevScWList

    copy_cached_input=False
    allow_alias=True

    input_will_use_promise=BasicP2EventProcessingSummary

    maxscw=None

    def main(self):
        self.thelist=[]
        l=self.input_scwlist.scwlistdata
        if self.maxscw is not None: l=l[:self.maxscw]
        for s in l:
            a=BinBackgroundSpectrumP2(assume=s)
            print(a,a.assumptions)
            self.thelist.append(a)


class EventFileList(da.DataAnalysis):
    input_scwlist=ddosa.RevScWList

    copy_cached_input=False
    allow_alias=True
    input_will_use_promise=BasicEventProcessingSummary

    maxscw=None

    def main(self):
        self.thelist=[]
        l=self.input_scwlist.scwlistdata
        if self.maxscw is not None: l=l[:self.maxscw]
        for s in l:
            a=ISGRIEvents(assume=s)
            print(a,a.assumptions)
            self.thelist.append(a)

class EventFileListP2(da.DataAnalysis):
    input_scwlist=ddosa.RevScWList

    copy_cached_input=False
    allow_alias=True
    input_will_use_promise=BasicP2EventProcessingSummary

    maxscw=None

    def main(self):
        self.thelist=[]
        l=self.input_scwlist.scwlistdata
        if self.maxscw is not None: l=l[:self.maxscw]
        for s in l:
            a=ISGRIEventsFinal(assume=s)
            print(a,a.assumptions)
            self.thelist.append(a)

class EventFileListP3(da.DataAnalysis):
    input_scwlist=ddosa.RevScWList

    copy_cached_input=False
    allow_alias=True
    input_will_use_promise=BasicP3EventProcessingSummary

    maxscw=None

    def main(self):
        self.thelist=[]
        l=self.input_scwlist.scwlistdata
        if self.maxscw is not None: l=l[:self.maxscw]
        for s in l:
            a=FineEnergyCorrection(assume=s)
            print(a,a.assumptions)
            self.thelist.append(a)

class EventFileListP2ScW(da.DataAnalysis):
    input_scwlist=ddosa.RevScWList

    copy_cached_input=False
    allow_alias=True
    input_will_use_promise=BasicP2EventProcessingSummary

    maxscw=None

    def main(self):
        self.thelist=[]
        l=self.input_scwlist.scwlistdata
        if self.maxscw is not None: l=l[:self.maxscw]
        for s in l:
            a=ISGRIEventsFinal(assume=s)
            print(a,a.assumptions)
            self.thelist.append([s,a])

class BinBackgroundMerged(ddosa.DataAnalysis):
    input_list=BinBackgroundList
    cached=True
    copy_cached_input=False

    # input_correction=?

    version="v3"

    tag=""


    def correct_energy_p3(self,d,t):
        return d

    def main(self):

        tcounts=0
        merged_1=None
        merged_2=None
        merged_3=None

        h1_pha1=None
        h1_pha2=None


        for i_a,a in enumerate(self.input_list.thelist):
            print("has",a,a.assumptions,id(a))
            print(dir(a))

            if hasattr(a,'nevents') and a.nevents==0:
                print("empty!")
                continue

            try:
                fn=getattr(a,'h2_ISGRI_PHA1_ISGRI_PI_1_1_'+self.tag).get_cached_path()
            except Exception as e:
                print("failed",e)
                continue # !!
            print("file name:",fn)
            f=getattr(a,'h2_ISGRI_PHA1_ISGRI_PI_1_1_'+self.tag).open()
            d=np.load(f)
            if merged_1 is None:
                merged_1=d[0]
            else:
                merged_1+=d[0]
            
            fn=getattr(a,'h2_ISGRI_PHA2_ISGRI_PI_1_1_'+self.tag).get_cached_path()
            print("file name:",fn)
            f=getattr(a,'h2_ISGRI_PHA2_ISGRI_PI_1_1_'+self.tag).open()
            d=np.load(f)
            if merged_2 is None:
                merged_2=d[0]
            else:
                merged_2+=d[0]
            
            fn=getattr(a,'h2_ISGRI_ENERGY_ISGRI_PI_1_1_'+self.tag).get_cached_path()
            print("file name:",fn)
            f=getattr(a,'h2_ISGRI_ENERGY_ISGRI_PI_1_1_'+self.tag).open()
            d=np.load(f)

            #t1,t2=a.input_scw.get_t()
            if hasattr(self,'input_scwlist'):
                scw=self.input_scwlist.scwlistdata[i_a]
                tc,dt=scw.get_t()
                d[0]=self.correct_energy_p3(d[0],tc)

            if merged_3 is None:
                merged_3=d[0]
            else:
                merged_3+=d[0]
            
            fn=getattr(a,'h1_pha1').get_cached_path()
            print("file name:",fn)
            f=getattr(a,'h1_pha1').open()
            d=np.load(f)
            #print d
            if h1_pha1 is None:
                h1_pha1=d
            else:
                h1_pha1[0]+=d[0]
            
            fn=getattr(a,'h1_pha2').get_cached_path()
            print("file name:",fn)
            f=getattr(a,'h1_pha2').open()
            d=np.load(f)
            #print d
            if h1_pha2 is None:
                h1_pha2=d
            else:
                h1_pha2[0]+=d[0]


        fn="h1_pha1.npy"
        np.save(fn,h1_pha1)
        np.savetxt(fn.replace(".npy",".txt"),np.column_stack((h1_pha1[1][:-1],h1_pha1[0])))
        self.h1_pha1=da.DataFile(fn)
        
        fn="h1_pha2.npy"
        np.save(fn,h1_pha2)
        np.savetxt(fn.replace(".npy",".txt"),np.column_stack((h1_pha2[1][:-1],h1_pha2[0])))
        self.h1_pha2=da.DataFile(fn)

        fn="merged_pha1_pi.fits"
        pyfits.PrimaryHDU(merged_1.astype(float)).writeto(fn,overwrite=True)
        self.h2_pha1_pi=da.DataFile(fn)

        fn="merged_pha2_pi.fits"
        pyfits.PrimaryHDU(merged_2.astype(float)).writeto(fn,overwrite=True)
        self.h2_pha2_pi=da.DataFile(fn)
        
        fn="merged_energy_pi.fits"
        pyfits.PrimaryHDU(merged_3.astype(float)).writeto(fn,overwrite=True)
        self.h2_energy_pi=da.DataFile(fn)
        
        print("total counts",tcounts)

        if self.plot:
            plot.p.clf()
            plot.p.plot(np.linspace(0,1024,2048),merged_3.sum(axis=1))
            plot.plot("energy_spectrum.png")

    plot=False

    def get_h2_pha1_pi(self):
        print("opening",self.h2_pha1_pi.path)
        return  pyfits.open(self.h2_pha1_pi.get_cached_path())[0].data

class CorrectBipar(ddosa.DataAnalysis):
    input_lut2=FinalizeLUT2
    input_background=BinBackgroundMerged
    #input_lut2=GenerateLUT2
    #input_detector_model=Fit3DModel
    #input_biparmodel=LocateBiparModel

    copy_cached_input=False

    bins="300"

    def get_version(self):
        v=self.get_signature()+"."+self.version
        if self.bins!="300":
            v+="."+self.bins
        return v

    def get_ebins(self):
        if self.bins=="2048":
            return 13+np.arange(2048)*0.4787
        if self.bins=="300":
            return np.logspace(1,np.log10(2048),300)
        if self.bins=="cube":
            f=pyfits.open("/Integral/data/resources/rmf_cubebins.fits")['EBOUNDS'].data
            return np.concatenate((f['E_MIN'],[f['E_MAX'][-1]]))
        
    def get_data(self):
        return [pyfits.open(self.input_background.h2_pha1_pi.get_path())[0].data]

    def reconstruct_bipar(self,data,lut2,ebins):

        dlut2_ph=np.zeros_like(lut2)
        dlut2_ph[1:-1,:]=(lut2[2:,:]-lut2[:-2,:])/2.
        dlut2_rt=np.zeros_like(lut2)
        dlut2_rt[:,1:-1]=(lut2[:,2:]-lut2[:,:-2])/2.
        
        corr=np.zeros((ebins.shape[0]-1,data.shape[1]))

        nsteps_ph=80
        nsteps_rt=1
        for dph in np.linspace(0,1,nsteps_ph):
            for drt in np.linspace(0,1,nsteps_rt):
                lut2_r=lut2+np.ones_like(lut2)*dph*dlut2_ph+np.ones_like(lut2)*drt*dlut2_rt
                #lut2_r=lut2+np.random.uniform(size=lut2.shape)*dlut2_ph
                corr += np.transpose(array([np.histogram(lut2_r[:,i],weights=data[:,i],bins=ebins)[0] for i in range(lut2.shape[1])]))/nsteps_ph/nsteps_rt

        return corr

    def main(self):
        lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())[0].data
        lut2[np.isnan(lut2)]=0
        
        ebins=self.get_ebins()
        ec=(ebins[:-1]+ebins[1:])*0.5

        for data in self.get_data():
    
            corr=self.reconstruct_bipar(data,lut2,ebins)
                
            fn="bipar_2d_corrected.fits"
            pyfits.HDUList( [pyfits.PrimaryHDU(corr),
                            pyfits.BinTableHDU.from_columns([
                                    pyfits.Column(name='EBOUND', format='E', array=ebins)
                            ])]).writeto(fn,overwrite=True)
            self.h2_energy_pi=da.DataFile(fn)
            
            np.savetxt("background_energy_1d.txt",np.column_stack((
                                                                ec,
                                                                corr[:,16:116].sum(axis=1),
                                                                corr[:,16:50].sum(axis=1),
                                                                corr[:,50:116].sum(axis=1),
                                                                corr[:,16:80].sum(axis=1)
                                                                )))
class CorrectBiparP4(CorrectBipar):
    input_lut2=FinalizeLUT2P4

class CorrectBipar2048(CorrectBipar):
    bins="2048"

class PredictLineBias(ddosa.DataAnalysis):
    input_lut2=FinalizeLUT2
    #input_lut2=GenerateLUT2
    input_detector_model=Fit3DModel
    input_biparmodel=LocateBiparModel

    copy_cached_input=False

    def main(self):
        pha_coord=np.arange(2049)

        #print "reading response.."
        #r3d=self.input_biparmodel.bipar_model.response3d(None,grouping=0.01)
        #r3d.np.loadfrom(self.input_lut2.response_3d.get_path())
        #print "reading response done"
        
        lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())[0].data

        det=self.input_detector_model.detector

        self.energies=[59.5,511.]
        for line_e in self.energies:
            #r=r3d.get_energy(line_e,disable_ai=False)
            r=self.input_biparmodel.bipar_model.generate_bipar(self.input_detector_model.detector,line_e)[0]

            pha_coord,rt_coord=np.meshgrid(np.arange(256),np.arange(2048))

            print(r.shape,lut2.shape,rt_coord.shape)

            m=(rt_coord>16.) & (rt_coord<116.)
            r[np.isnan(r)]=0
            lut2[np.isnan(lut2)]=0
            
            ebins=np.linspace(0,1024,2049)
            ec=(ebins[:-1]+ebins[1:])*0.5

            

            #h=np.histogram(lut2,weights=r,bins=ebins)
            #np.savetxt("line_energy_%.5lg_1d.txt"%line_e,np.column_stack(((h[1][1:]+h[1][:-1])*0.5,h[0])))

            corr=np.transpose(array([np.histogram(lut1[:,i],weights=r[:,i],bins=ebins)[0] for i in range(lut2.shape[1])]))
            pyfits.PrimaryHDU(corr).writeto("line_2d_%.5lgcorrected.fits"%line_e,overwrite=True)
            
            for rt1,rt2 in ((16,50),(50,116),(16,116)):
                espec=corr[:,rt1:rt2].sum(axis=1)
                av=sum(espec*ec)/sum(espec)
                print(rt1,rt2,"np.average energy",av,"bias",av-line_e)

            np.savetxt("line_energy_%.5lg_1d.txt"%line_e,np.column_stack((
                                                                    ec,
                                                                    corr[:,16:116].sum(axis=1),
                                                                    corr[:,16:50].sum(axis=1),
                                                                    corr[:,50:116].sum(axis=1),
                                                                    corr[:,16:80].sum(axis=1)
                                                                    )))

            #setattr(self,'line_%.5lg',da.DataFile(fn))

class LinesModel1D(ddosa.DataAnalysis):
    #input_lut2=GenerateLUT2
    input_detector_model=Fit3DModel

    copy_cached_input=False

    def main(self):
        pha_coord=np.arange(2049)

 #       print "reading response.."
 #       r3d=bipar_model.response3d(None,grouping=0.01)
 #       r3d.np.loadfrom(self.input_lut2.response_3d.get_path())
 #       print "reading response done"

        det=self.input_detector_model.detector

        self.energies=[59.,511.]
        for line_e in self.energies:
            #r=r3d.get_energy(line_e,disable_ai=False)
            r=self.input_biparmodel.bipar_model.generate_bipar(bipar_model.detector(),line_e)[0]

            r1d_50=r[:,:50].sum(axis=1)

            region_mask=r1d_50>(r1d_50/100.)
            fn="line_%.5lg_1d.txt"%line_e
            np.savetxt(fn,np.column_stack((pha_coord[region_mask],r1d_50[region_mask])))

            setattr(self,'line_%.5lg',da.DataFile(fn))
            
            
class CompareLines1D(ddosa.DataAnalysis):
    input_model=LinesModel1D
    input_background=BinBackgroundMerged
    copy_cached_input=False

    def main(self):
        data=pyfits.open(self.input_background.h2_pha1_pi.get_path())[0].data
        data1d_50=data[:,:50].sum(axis=1)
        pha_coord=np.arange(2049)

        print(data1d_50.shape)

        for line_e in self.input_model.energies:
            r1d_50=r[:,:50].sum(axis=1)

            setattr(self,'line_%.5lg',da.DataFile(fn))

            np.savetxt("line_%.5lg_1d.txt"%line_e,np.column_stack((pha_coord[region_mask],data1d_50[region_mask],r1d_50[region_mask])))


class BinBackgroundMergedP2(BinBackgroundMerged):
    input_list=BinBackgroundListP2
    
    tag="P2"


    
class Fit3DModelRev(ddosa.DataAnalysis):
    #input_rev=ddosa.Revolution #????
    #rev=ddosa.Revolution

    allow_alias=False
    run_for_hashe=True

    def main(self):
        return Fit3DModel(assume=[
            Bipar(use_bipar=BinBackgroundMerged),
            BinBackgroundList(input_scwlist=ddosa.RevScWList),
            ddosa.RevScWList(input_rev=Revolution)],
            use_mutating=False
        )



class GenerateLUT2(ddosa.DataAnalysis):
    #input_p=FindPeaks
    input_detector_model=Fit3DModelRev
    input_biparmodel=LocateBiparModel

    #version="v23_hr_nolog_lc_3d"
    version="v23_hr_nolog_lc_3d_medmode_mm2"##

   # cache=cache_local
    cached=True

    # config fields
    generate_lut2_3d=False
    resolution_factor=0.1
    resolution_step=1 #!!
    response_grouping=0.01

    writer3d=True
    mcarf=False
    ltmodel=False

    watched_analysis=True

    lut2_generator="v3"
    #lut2_generator="simple_faster"

    def get_version(self):
        return self.get_signature()+"."+self.version+\
                    (".lut23d" if self.generate_lut2_3d else ".lut22d")+\
                    ".rs%i"%self.resolution_factor+\
                    ".rg%.5lg"%self.response_grouping+\
                    (".rstep%.5lg"%self.resolution_step if self.resolution_step!=1 else "")+\
                    (".3dr" if self.writer3d else "")+\
                    (".mcarf" if self.mcarf else "")+\
                    ("."+self.lut2_generator if self.lut2_generator!="simple_faster" else "")+\
                    (".ltmodel" if self.ltmodel else "")

    def lut2_1d_to_3d(self):
        print("converting from 1d lut2")
        lut2=pyfits.open(self.lut2_1d.get_path())[0].data
        #from scipy.ndimage import gaussian_filter
        #lut2=gaussian_filter(lut2,1)
        fd=pyfits.open(os.environ['CURRENT_IC']+"/ic/ibis/mod/isgr_3dl2_mod_0001.fits")
        for i in range(500):
            fd[1].data[i]=lut2.np.transpose()[:,::2]*30.
        fd.writeto("lut2_3d.fits",overwrite=True)
        return "lut2_3d.fits"


    def main(self):
        #det=bipar_model.estimate_parameters_from_peaks(self.input_p.max1,self.input_p.max2) 
        #print det

        det=self.input_detector_model.detector
        self.detector=det

        energy=10
        energies=[]
        ei=1
        while energy<1500:
            energies.append(energy)
            energy+=(0.05+(ei/100.)**2/23.1827)*0.5*self.resolution_factor
            ei+=1

        energies=array(energies)

        print(energies,energies[:-1]-energies[1:])

        source_model_pure=lambda x:x**-2
        if self.mcarf:
            d=pyfits.open("/Integral/data/resources/arfs/arf_mc_oldenergies.fits")[1].data
            arf=interp1d((d['ENERG_LO']+d['ENERG_HI'])/2.,d['SPECRESP'],bounds_error=False)
            source_model_arf=lambda x:(source_model_pure(x)*arf(x))
        else:
            source_model_arf=lambda x:source_model_pure(x)

        if self.ltmodel:
            from scipy.special import erf
            source_model_final=lambda x:(source_model_arf(x)*erf(x/40))
        else:
            source_model_final=lambda x:source_model_arf(x)
    

        genfunc=getattr(self.input_biparmodel.bipar_model,"make_lut2_"+self.lut2_generator)
        lut2=genfunc(detector_model=det,
                     resolution_step=self.resolution_step,
                     write3dresp=self.writer3d,
                     energies=energies,
                     writelut23d=self.generate_lut2_3d,
                     logenergies=False,
                     render_model="m0",
                     group_energies=self.response_grouping,
                     source_model=source_model_final)[0]

       # bipar_model.response_response()

     #   self.response_2d=DataFile("response_2d.fits")
     #   self.response_2d=DataFile("response_2d_40.fits")
     #   self.response_2d=DataFile("response_2d_100.fits")
        #self.response_2d=DataFile("response_2d_116.fits")
        self.lut2_1d=DataFile("lut2_1d.fits")

        if self.generate_lut2_3d:
            self.lut2_3d=DataFile("lut2_3d.fits")
       # self.line60=DataFile("line2d_60.fits")
        #self.line511=DataFile("line2d_511.fits")

        if self.writer3d:
            self.response_3d=DataFile("response_3d.fits")

        #os.system("rm -fv response_3d.fits")

class ReferenceModelSet(ddosa.DataAnalysis):
    allow_alias=True
    run_for_hashe=True

    def main(self):
        reference_revs=["1626","1927"]

        self.models=[]
        for rev in reference_revs:
            self.models.append([rev,Fit3DModelRev(assume=[ddosa.Revolution(input_revid=rev)])])

class InterpolatedDetectorModel(ddosa.DataAnalysis):
    input_rev=ddosa.Revolution

    input_reference=ReferenceModelSet

    def main(self):
        ijd0=self.input_rev.get_ijd()
        revid=float(self.input_rev.input_revid.str())

        models=[]
        for rrev,model in self.input_reference.models:
            print((rrev,model))
            print((model.detector))
            models.append(dict([(k,getattr(model.detector,k)) for k in ['mu_e','mu_t','tau_e','tau_t','offset']]))

        modelset=pd.DataFrame(models)

        self.detector=self.input_biparmodel.bipar_model.detector()
        self.detector.mu_e=modelset.mu_e

        raise NotImplemented()


class GenerateModelledLUT2(GenerateLUT2):
    input_detector_model=InterpolatedDetectorModel

class CubeBins:
    def __init__(self):
        if 'ISGRI_RMF_256' in os.environ:
            isgri_rmf=pyfits.open(os.environ["ISGRI_RMF_256"])
        else:
            isgri_rmf=pyfits.open(os.environ["INTEGRAL_DATA"]+"/resources/rmf_256bins.fits")

        self.emin,self.emax=(lambda x:(x['E_MIN'],x['E_MAX']))(isgri_rmf['EBOUNDS'].data)
        self.energies=(self.emin+self.emax)/2.
        self.denergies=(self.emax-self.emin)/2.
        self.pha=self.energies*2

class FitLocalLinesRevCorrected(da.DataAnalysis):
    pass

class FitLocalLinesRevCorrectedP4(da.DataAnalysis):
    pass

class BadLineFit(da.AnalysisException):
    pass

class VerifyLines(ddosa.DataAnalysis):
    input_correctedlines=FitLocalLinesRevCorrected

    factor=3
    syst_percent=5

    copy_cached_input=False

    def main(self):
        he_line=self.input_correctedlines.he_line_fullrt
        print("HE line",he_line)
        x0,x1,x2=he_line['centroid'],he_line['x0_lower_limit'],he_line['x0_upper_limit']
        dx=(x2-x1)/2.
        if abs(x0-511.)>dx*self.factor+self.syst_percent*511./100.:
            raise BadLineFit()
        print("decent line fit",x0,dx)
        
        le_line=self.input_correctedlines.le_line_fullrt
        print("LE line",le_line)
        x0,x1,x2=le_line['centroid'],le_line['x0_lower_limit'],le_line['x0_upper_limit']
        dx=(x2-x1)/2.
      #  if abs(x0-59.3)>dx*self.factor:
      #      raise BadLineFit()
      #  print "decent line fit",x0,dx

class VerifyLinesP4(VerifyLines):
    input_correctedlines=FitLocalLinesRevCorrectedP4

class FinalizeLUT2(ddosa.DataAnalysis):
    input_lut2=GenerateLUT2
    input_rev=Revolution

    #input_finelinecorr=FitLocalLinesRevCorrected

    copy_cached_input=False

    cached=True

    def get_version(self):
        v=self.get_signature()+".x"
        if self.corr is None:
            return v+"vbase5.2"

        if self.corr!="pb3":
            return v+"corr"+self.corr

        return v+"_corr"+self.corr+".%.5lg"%self.corr_par

    corr=None
    corr_par=1.

    def interpolate(self,ph,ph1,l1,dl1,ph2,l2,dl2):
        return l1+(l2-l1)/(ph2-ph1)*(ph-ph1)
    
    def interpolate_poly(self,ph,ph1,l1,dl1,ph2,l2,dl2):
     #   return l1+(l2-l1)/(ph2-ph1)*(ph-ph1)
        from numpy.linalg import solve
        
        matrix=[[1,ph1,ph1**2,ph1**3],
                [1,ph2,ph2**2,ph2**3],
                [0,1,2*ph1,3*ph1**2],
                [0,1,2*ph2,3*ph2**2]]
        F0=[l1,l2,dl1,dl2]

        a,b,c,d=solve(matrix,F0)

        f=lambda x:a+b*x+c*x**2+d*x**3
        df=lambda x:b+2*c*x+3*d*x**2


        return a+b*ph+c*ph**2+d*ph**3

    def main(self):

        orig_lut2_filename = self.input_lut2.lut2_1d.get_path()

        orig_lut2 = pyfits.open(orig_lut2_filename)[0].data

        rt, ph = np.meshgrid(np.arange(256), np.arange(2048))
        if orig_lut2.shape[0]==2048:
            lut2=orig_lut2

            print(ph.shape,rt.shape)
            for _ph in range(2048):
                badrt = np.isnan(lut2[_ph,:]) & (rt[_ph,:]<30)
                if np.sum(~badrt)==0:
                    lut2[_ph,:]=0
                    continue

                minrt = rt[_ph,~badrt][0]+1 # !!

                maxrt = 150

                lut2[_ph,:minrt] = lut2[_ph,minrt]
                lut2[_ph,maxrt:] = np.NaN# lut2[_ph,maxrt]

            lut2[:15,:]=0

            t = self.input_rev.get_ijd()
        elif orig_lut2.shape[0]==256:
            lut2_gain = pyfits.open(orig_lut2_filename)[1].data

            cb = CubeBins()

            lut2 = rt*ph*0.
            
            for _rt in range(256):
                m = ~np.isnan(orig_lut2[:,_rt])

                lut2_row_good = orig_lut2[m,_rt]
                lut2gain_row_good = lut2_gain[m,_rt]
                pha_good = cb.pha[m]

                for _ph in range(2048):
                    if len(pha_good) == 0:
                        lut2[_ph,_rt] = 0
                    elif _ph<pha_good[0]:
                        lut2[_ph,_rt] = lut2_row_good[0]-(pha_good[0]-_ph)/lut2gain_row_good[0]
                    else:
                        i_good = np.where(_ph>=pha_good)[0][-1]
                        if i_good == len(pha_good)-1:
                            lut2[_ph,_rt] = 0
                        else:
                            lut2[_ph,_rt] = self.interpolate_poly(_ph,
                                                pha_good[i_good],lut2_row_good[i_good],1./lut2gain_row_good[i_good],
                                                pha_good[i_good+1],lut2_row_good[i_good+1],1/lut2gain_row_good[i_good+1]
                                                )
                                
                            #gain=(lut2_row_good[i_good+1]-lut2_row_good[i_good])/(pha_good[i_good+1]-pha_good[i_good])
                            
                            #print  "at",_rt,_ph,gain,lut2[_ph,_rt]
                    

            for _ph in range(2048):
                badrt = np.isnan(lut2[_ph,:]) & (rt[_ph,:]<30)
                if sum(~badrt)==0:
                    lut2[_ph,:]=0
                    continue

                minrt = rt[_ph,~badrt][0] + 1 # !!

                maxrt = 150

                lut2[_ph,:minrt] = lut2[_ph,minrt]
                lut2[_ph,maxrt:] = np.NaN# lut2[_ph,maxrt]

            lut2[:15,:]=0

        else:
            raise Exception("strange LUT2 shape")

        t=self.input_rev.get_ijd()
        if self.corr=="pb2":
            l2f=lut2.flatten()
            m=l2f<59.
            l2f[m]=(l2f+((l2f-59.)/59.)**2*6)[m]
            lut2=l2f.reshape(lut2.shape)


        if self.corr=="pb3":
            def rf(en):
                return en-self.corr_par*10./(1.+(en/59)**2)

            def f(en):
                return rf(en)-rf(59)+59

            l2f = lut2.flatten()
            l2f = f(l2f)
            lut2 = l2f.reshape(lut2.shape)

        l2f = lut2.flatten()
        l2f = self.fine_correction(l2f)
        lut2 = l2f.reshape(lut2.shape)

        nf = "lut2_1d_final.fits"

        header=pyfits.Header()
        for field in ('mu_e','mu_t','tau_e','tau_t','gain','offset','rt_offset','V'):
            header[field[:8]]=getattr(self.input_lut2.detector,field)

        print("setting header",header)

        pyfits.PrimaryHDU(lut2,header=header).writeto(nf,overwrite=True)

        self.lut2_1d=da.DataFile(nf)

    def fine_correction(self,en):
        return en

    def lut2_1d_to_3d(self):
        # copied from GLT amd file!
        print("converting from 1d lut2")
        lut2=pyfits.open(self.lut2_1d.get_path())[0].data
        #from scipy.ndimage import gaussian_filter
        #lut2=gaussian_filter(lut2,1)
        fd=pyfits.open(os.environ['CURRENT_IC']+"/ic/ibis/mod/isgr_3dl2_mod_0001.fits")
        for i in range(500):
            fd[1].data[i]=lut2.np.transpose()[:,::2]*30.
        fd.writeto("lut2_3d.fits",overwrite=True)
        return "lut2_3d.fits"

class FinalizeLUT2P4(FinalizeLUT2):
    input_finelinecorr=FitLocalLinesRevCorrected

    p4origin="centroid"
    p4le="v4"
    
    def get_version(self):
        v=FinalizeLUT2.get_version(self)

        return v+"."+self.p4origin+".le"+self.p4le

    def fine_correction(self,en):
        lines=pd.read_csv(getattr(self.input_finelinecorr,'local_lines_fullrt_fn').open(), delim_whitespace=True)
        print(lines)

        le_x0 = lines.centroid.iloc[0]
        he_x0 = lines.centroid.iloc[1]
        #le_x0 = lines.bestfit_x0.iloc[0]
        #he_x0 = lines.bestfit_x0.iloc[1]
        le_model=59.
        he_model=511.

        print(("applying final fine post correction:", 1./(he_x0-le_x0)*(he_model-le_model), le_model-le_x0))

        vle=30
        new_vle=25


        def rf(_en,tuned_par):
            return _en-self.corr_par*tuned_par*10./(1.+(_en/59)**2)
        
        def transform_norf(_en):
            return le_model+(_en-le_x0)/(he_x0-le_x0)*(he_model-le_model)

        def transform(_en,tuned_par):
            return le_model+(rf(_en,tuned_par)-rf(le_x0,tuned_par))/(rf(he_x0,tuned_par)-rf(le_x0,tuned_par))*(he_model-le_model)

        tuned_par_fitted=min([[abs(rf(vle,tuned_par)-new_vle),tuned_par] for tuned_par in np.logspace(-1,1,100)])[1]

        print(("tuned par",tuned_par_fitted))

        #def transform(_en):
        #    return f(transform_fixlines(_en))

        print(("transform",vle,"=>",transform(vle,tuned_par_fitted)))
        print(("transform",le_x0,"=>",transform(le_x0,tuned_par_fitted)))
        print(("transform",he_x0,"=>",transform(he_x0,tuned_par_fitted)))
        print(("transform no rf",vle,"=>",transform_norf(vle)))
        print(("transform no rf",le_x0,"=>",transform_norf(le_x0)))
        print(("transform no rf",he_x0,"=>",transform_norf(he_x0)))

        new_en=transform(en,tuned_par_fitted)

        return new_en

class ISGRI_RISE_MOD(ddosa.DataAnalysis):
    input_lut2=FinalizeLUT2
    input_rev=Revolution

    cached=True

    def main(self):
        import resttimesystem as ts
        out_fn="isgr_rise_mod_%.4i.fits"%int(self.input_rev.input_revid.str()) #//

        dc=pilton.heatool("dal_create")
        dc["obj_name"]=out_fn
        dc["template"]="ISGR-RISE-MOD.tpl"
        remove_withtemplate(dc["obj_name"].value+"("+dc["template"].value+")")
        dc.run()

        val_start_ijd,val_stop_ijd=list(map(float,ts.converttime("REVNUM",self.input_rev.input_revid.str(),"IJD").split()[1:]))
        val_start_utc=ts.converttime("IJD",val_start_ijd,"UTC")
        val_stop_utc=ts.converttime("IJD",val_stop_ijd,"UTC")

        f=pyfits.open(out_fn)
        d=pyfits.open(self.input_lut2.lut2_1d.get_path())[0]

        f[1].data=np.zeros(2048,dtype=f[1].data.dtype)
        f[1].data['CHANNEL']=np.arange(2048)
        f[1].data['ENERGY']=d.data[:,30]
        f[1].data['CORR']=d.data[:,:]/np.outer(d.data[:,30],np.ones(256))

        f[1].header['ORIGIN']="ISDC"
        f[1].header['VERSION']=1
        f[1].header['FILENAME']=out_fn
        f[1].header['LOCATN']=out_fn
        f[1].header['RESPONSI']="Volodymyr Savchenko"
        f[1].header['STRT_VAL']=val_start_utc
        f[1].header['END_VAL']=val_stop_utc
        f[1].header['VSTART']=val_start_ijd
        f[1].header['VSTOP']=val_stop_ijd
        #f.writeto("/sps/integral/data/ic/ic_snapshot_20140321/ic/ibis/mod/isgr_rise_mod_0001.fits",overwrite=True)
        f.writeto(out_fn,overwrite=True)

        self.isgr_rise_mod=DataFile(out_fn)

class ISGRI_RISE_MOD_IDX(ddosa.DataAnalysis):
    def main(self):
        dc=pilton.heatool("dal_create")
        dc["obj_name"]="/sps/integral/data/ic/ic_snapshot_20140321/idx/ic/ISGR-RISE-MOD-IDX.fits"
        dc["template"]="ISGR-RISE-MOD-IDX.tpl"
        remove_withtemplate(dc["obj_name"].value+"("+dc["template"].value+")")
        dc.run()

        da=pilton.heatool("dal_attach")
        da['Parent']="/sps/integral/data/ic/ic_snapshot_20140321/idx/ic/ISGR-RISE-MOD-IDX.fits"
        da['Child1']="/sps/integral/data/ic/ic_snapshot_20140321/ic/ibis/mod/isgr_rise_mod_0001.fits"
        da.run()

        f=pyfits.open(da['Parent'].value)
        f[1].data[0]['VERSION']=1
        f[1].data[0]['VSTART']=0
        f[1].data[0]['VSTOP']=300000
        f.writeto(da['Parent'].value,overwrite=True)

        dv=pilton.heatool("dal_verify")
        dv["indol"]="/sps/integral/data/ic/ic_snapshot_20140321/idx/ic/ISGR-RISE-MOD-IDX.fits"
        dv['checksums']="yes"
        dv['backpointers']="yes"
        dv['detachother']="yes"
        dv.run()

        da=pilton.heatool("dal_attach")
        da['Parent']="/sps/integral/data/ic/ic_snapshot_20140321/idx/ic/ic_master_file.fits"
        da['Child1']="/sps/integral/data/ic/ic_snapshot_20140321/idx/ic/ISGR-RISE-MOD-IDX.fits"
        da.run()

        pass


class LineBipars(ddosa.DataAnalysis):
    input_lut2=FinalizeLUT2
    input_biparmodel=LocateBiparModel
    input_fit3dmodel=Fit3DModel

    #input_background=BinBackgroundMerged
    input_corrdata=CorrectBipar2048

    set_energies="two"
    #set_energies=None

    cached=True

    def get_version(self):
        v=self.get_signature()+"."+self.version
        if self.set_energies is not None:
            v+="ene"+str(self.set_energies) # or list
        return v
                
    def main(self):
        import ibismm
        IBISMM=ibismm.IBISMM() # yes yes
        IBISMM.init_response()

        lut2_fn=self.input_lut2.lut2_1d.get_path()

        #energies=[511]
        #energies=[22]
        #energies=[511]
        #energies=[59.3182, 511]
        #energies=[22, 59.3182, 150, 511]
        
        if self.set_energies=="two":
            self.energies=[50.]
            #self.energies=[59.3182, 511]
        elif self.set_energies=="allmany":
            self.energies=[22,30,57.9817,  59.3182,  67.2443,  72.8042,   74.9694, 84,45, 84.936, 87,32, 150, 511]
        else:
            pass

        lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())[0].data
        lut2[np.isnan(lut2)]=0
        
        ebins=self.input_corrdata.get_ebins()
        ec=(ebins[:-1]+ebins[1:])*0.5
        de=(-ebins[:-1]+ebins[1:])*0.5

        self.ebins=ebins
        self.ec=ec
        self.de=de

        data=self.input_corrdata.get_data()[0]  
        corrdata=self.input_corrdata.reconstruct_bipar(data,lut2,ebins)
            
        det=self.input_fit3dmodel.detector
        #det.rt_offset-=2.
        
        fn="data.fits"
        pyfits.HDUList([pyfits.PrimaryHDU(data),pyfits.ImageHDU(corrdata)]).writeto(fn,overwrite=True)
        self.data=da.DataFile(fn)

        for line_energy in self.energies:

            from timeit import default_timer as timer

            start = timer()

            line_model_ibismm,line_model_jrec=IBISMM.get_line_model(line_energy,lut2_fn)

            end1 = timer()

            line_model=self.input_biparmodel.bipar_model.make_bipar_monoenergetic(det,line_energy,resolution_step=1,resolutionfactor=1.,render_model="m0")
            #line_model,rt_bip,q_bip,intensity=self.input_biparmodel.bipar_model.generate_bipar(det,line_energy,render_model="m0")

            #fn="line_%.5lg.txt"%line_energy
            #np.savetxt(fn,np.column_stack((rt_bip,q_bip,intensity)))

            end2 = timer()
            
            print("Java:",end1 - start, "Python-C", end2-end1)

            corrbipibismm=self.input_corrdata.reconstruct_bipar(line_model_ibismm,lut2,ebins)
            corrbip=self.input_corrdata.reconstruct_bipar(line_model,lut2,ebins)

            fn="line_%.5lg.fits"%line_energy
            setattr(self,fn.replace(".fits",""),da.DataFile(fn))
            pyfits.HDUList([pyfits.PrimaryHDU(line_model_ibismm),pyfits.ImageHDU(line_model),pyfits.ImageHDU(corrbipibismm),pyfits.ImageHDU(corrbip),pyfits.ImageHDU(line_model_jrec)]).writeto(fn,overwrite=True)

dcenter=lambda x,y:sum(x*y)/sum(y)
dwidth=lambda x,y:(sum(x*x*y)/sum(y)-dcenter(x,y)**2)**0.5

class LineModel:
    def __init__(self):
        pass

    pb_norm=[1,1]
    w_norm=[1,1]
    bi_norm=[1]

    gw_model=(0,1,0,0,0)
        
    e0=60.
    w0=e0*0.06
    
    def en2g(self,energy):
        gain2,gain,offset,widthgain,widthoffset=self.gw_model
        return gain+gain2*(energy-self.e0)/self.e0
    
    def en2ch(self,energy):
        gain2,gain,offset,widthgain,widthoffset=self.gw_model
        return self.en2g(energy)*(energy-self.e0)+self.e0+offset
    
    def en2w(self,energy):
        gain2,gain,offset,widthgain,widthoffset=self.gw_model
        return widthgain*(energy-self.e0)+self.w0+widthoffset
    
    def get_gaussian(self,line_energy):

        ec=self.ec
        de=self.de

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        return gaussian(ec,self.en2ch(line_energy),self.en2w(line_energy))
                           

    def get_pb(self):
        pb_model_a=self.get_gaussian(74.9694)+0.6*self.get_gaussian(72.804) # k-alpha
        pb_model_b=0.23*self.get_gaussian(84.936)+0.12*self.get_gaussian(84.450) + 0.08*self.get_gaussian(87.320)
        return pb_model_a,pb_model_b

    def get_w(self):
        w_model_a=self.get_gaussian(59.3182)+0.58*self.get_gaussian(57.981) 
        w_model_b=0.11*self.get_gaussian(66.951) +0.22*self.get_gaussian(67.244+1.) +0.08*self.get_gaussian(69.067+1.) #!!!
        return w_model_a,w_model_b
    
    def get_bi(self):
        bi_model=self.get_gaussian(77.108)+0.6*self.get_gaussian(74.8148)
        bi_model+=0.12*self.get_gaussian(86.834) +0.23*self.get_gaussian(87.343) +0.09*self.get_gaussian(89.830)
        return [bi_model]

    def get_1d(self,comp=None):
        pb=self.get_pb()[0]*self.pb_norm[0]+self.get_pb()[1]*self.pb_norm[1]
        w=self.get_w()[0]*self.w_norm[0]+self.get_w()[1]*self.w_norm[1]
        bi=self.get_bi()[0]*self.bi_norm[0]

        if comp is None:
            return pb+w+bi

        elif comp==0:
            return pb
        elif comp==1:
            return w
        elif comp==2:
            return bi


    def get_x(self):
        return list(self.gw_model)+self.pb_norm+self.w_norm+self.bi_norm
    
    def get_xmin(self):
        return [-0.5,0.9,-10,0,-2]+[0.01,0.01,0.01,0.01,0.001]

    def get_xmax(self):
        return [0.5,1.5,10,1,20]+[100,100,100,100,100]
    
    def set_x(self,x):
        self.gw_model=x[:5]
        self.pb_norm=x[5:7]
        self.w_norm=x[7:9]
        self.bi_norm=[x[9]]
        
    def get_mask(self):
        a,b=self.get_pb()
        c,d=self.get_w()
        return (a>a.max()/100.) | (b>b.max()/100.)| (c>c.max()/100.)| (d>d.max()/100.)

    def show(self):
        print("gain,offset",self.gw_model[:3])
        print("widthgain,widthoffset",self.gw_model[3:5])
        print("norms",self.pb_norm,self.w_norm,self.bi_norm)
        print("examples",[[en,self.en2ch(en),self.en2g(en),self.en2w(en)] for en in [60,74,84]])

class FitLineBipars(ddosa.DataAnalysis):
    input_linebipars=LineBipars

    copy_cached_input=False

    def get_data(self):
        if self.data is None:
            print("reading data")
            self.data=[f.data for f in pyfits.open(self.input_linebipars.data.get_path())]
        return self.data
    
    line_models={}
    data=None

    def get_model(self,line_energy):
        if line_energy not in self.line_models:
            an="line_%.5lg"%line_energy
            print("reading",an)
            self.line_models[line_energy]=[f.data for f in pyfits.open(getattr(self.input_linebipars,an).get_path())]
        return self.line_models[line_energy]
    
    def fit_model_complex(self,ec,de,model,data,key=None):
        model.ec=ec
        model.de=de

        summodel=model.get_1d()

        m=model.get_mask()

        print(np.where(m))

        err=data**0.5

        background_0=0
        background_1=0
        
        self.global_i=0
           
        def region_model(x,p,comp=None):
            model.set_x(p[:-2])
            model.ec=x

            b0,b1=p[-2:]

            rm=(model.get_1d(comp)/mscale+b0+b1*(x-e_scale)/e_scale)*dscale
            #print "model components:",nanmax((norm*model_offset))/mscale,b0,b1*((x-e_scale)/e_scale).max(),"and",nanmax(rm),x[argmax(rm)],"data",data[m].max()/dscale
            return rm

        def residual_func(region_model,p):

            r=((data[m]-region_model(ec[m],p))/err[m])**2
            rs=np.nansum(r)/np.nansum(r/r)
            if self.global_i%100==0:
                print(p,rs)
             #   np.savetxt("fit_model_"+str(self.global_i)+"_"+str(key)+".txt",np.column_stack((ec[m],de[m],region_model(ec[m],p),data[m],r)))
            #print self.global_i,"for",p,r,rs
            self.global_i+=1
            return rs
        
        x0=model.get_x()+[0.5,0]
        xmin=model.get_xmin()+[0.1,-5]
        xmax=model.get_xmax()+[10,5]

        print(len(x0))
        print(model.get_xmin())
        print(model.get_xmax())

        scale=1
        e_scale=np.average(ec[m])

        mc=data[m].argmax()
        dscale=data[m][mc]
        mscale=summodel[m].max()
        
        print("scale",dscale,mscale,e_scale)

        import nlopt
        opt = nlopt.opt(nlopt.LN_COBYLA, len(x0))
        opt.set_lower_bounds(xmin)
        opt.set_upper_bounds(xmax)
        opt.set_min_objective(lambda p,g:residual_func(region_model,p))
        opt.set_xtol_rel(1e-4)
    
        x = opt.optimize(x0)
        optf = opt.last_optimum_value()

        fitted_model=region_model(ec,x)
        #x_bkg=copy(x)
        #x_bkg[:-4]=0
        #fitted_model_bkg=region_model(ec,x_bkg)

        np.savetxt("fit_model_"+str(key)+".txt",np.column_stack([ec,de,fitted_model,data,region_model(ec,x,0),region_model(ec,x,1),region_model(ec,x,2)])[m,:])
        #np.savetxt("fit_model_"+str(key)+".txt",np.column_stack([ec,de,model,fitted_model,fitted_model_bkg,data]+[a*n/mscale*dscale for a,n in zip(x[:-4],models)])[m,:])

        #x[:-4]*=dscale/mscale

        print("fitted:",x)
        print(model.show())

        return x,fitted_model #,fitted_model_bkg


    def fit_model(self,ec,de,models,data,key=None):
        summodel=reduce(lambda x,y:x+y,models)


        m=np.zeros_like(summodel,dtype=bool)
        for model in models:
            m=m | (model>model.max()/1000.)

        print(np.where(m))

        err=data**0.5

        norm=1
        offset=0
        background_0=0
        background_1=0
        
        self.global_i=0
           
        def region_model(x,p):
            offset,stretch,b0,b1=p[-4:]
            norms=p[:-4]


            rm=(summodel_offset+b0+b1*(x-e_scale)/e_scale)*dscale
            #print "model components:",nanmax((norm*model_offset))/mscale,b0,b1*((x-e_scale)/e_scale).max(),"and",nanmax(rm),x[argmax(rm)],"data",data[m].max()/dscale
            return rm

        
        def residual_func(region_model,p):
            r=((data[m]-region_model(ec[m],p))/err[m])**2
            rs=np.nansum(r)/np.nansum(r/r)
            if self.global_i%100==0:
                print(p,rs)
             #   np.savetxt("fit_model_"+str(self.global_i)+"_"+str(key)+".txt",np.column_stack((ec[m],de[m],region_model(ec[m],p),data[m],r)))
            self.global_i+=1
            return rs
        
        nmodels=len(models)
        x0=[0.5]*nmodels+[0,1,0.5,0]

        scale=1
        e_scale=np.average(ec[m])

        mc=data[m].argmax()
        dscale=data[m][mc]
        mscale=summodel[m].max()
        
        print("scale",dscale,mscale,e_scale)

        import nlopt
        opt = nlopt.opt(nlopt.LN_COBYLA, len(x0))
        opt.set_lower_bounds([0.001]*nmodels+[-0.05,0.2,0.1,-5])
        opt.set_upper_bounds([10]*nmodels+[0.05,10,10,5])
        opt.set_min_objective(lambda p,g:residual_func(region_model,p))
        opt.set_xtol_rel(1e-4)
    
        x = opt.optimize(x0)
        optf = opt.last_optimum_value()

        fitted_model=region_model(ec,x)
        x_bkg=copy(x)
        x_bkg[:-4]=0
        fitted_model_bkg=region_model(ec,x_bkg)

        np.savetxt("fit_model_"+str(key)+".txt",np.column_stack([ec,de,model,fitted_model,fitted_model_bkg,data]+[a*n/mscale*dscale for a,n in zip(x[:-4],models)])[m,:])

        x[:-4]*=dscale/mscale

        print("fitted:",x)

        return x,fitted_model,fitted_model_bkg


    plot=False
    #plot=True


    def get_gaussian(self,line_energy):
        ec=self.input_linebipars.ec
        de=self.input_linebipars.de

        if False:
            try:
                corrbip=self.get_model(line_energy)[2]
                center_base=dcenter(ec,corrbip[:,16:30].sum(1))
                width_base=dwidth(ec,corrbip[:,16:30].sum(1))
            except:
                pass
        else:
            center_base=line_energy
            width_base=center_base*0.06 #!!

        print("basic:",width_base,center_base)

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        return gaussian(ec,center_base,width_base)
    
    def get_gaussian_line_model(self,line_energy,gw_model):
        gain2,gain,offset,widthgain,widthoffset=gw_model

        ec=self.input_linebipars.ec
        de=self.input_linebipars.de

        def gaussian(x, mu, sig):
            return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.)))

        return gaussian(ec,gain*line_energy+offset,widthgain*line_energy+widthoffset)

    def fit_by_rt(self,fmodel):
        ec=self.input_linebipars.ec
        de=self.input_linebipars.de

 #       rtb=np.arange(14,116)[::1]
        rtb=np.arange(14,74)[::30]
        columns=[ec,de]

        fit_results=[]

        corrdata=self.get_data()[1]

        for rt1, rt2 in zip(rtb[:-1],rtb[1:]):
            model=fmodel()

            data_1d=corrdata[:,rt1:rt2].sum(1)
            
        #    p,fitted_model_1d,fitted_model_bkg=self.fit_model(ec,de,model_1d,data_1d,key="%.5lg_%.5lg"%(rt1,rt2)) #,key="%.5lg_%.5lg_%.gl"%(rt1,rt2,line_energy))
            #m=model_1d>model_1d.max()/100.

            p,fitted_model_1d=self.fit_model_complex(ec,de,model,data_1d,key="%.5lg_%.5lg"%(rt1,rt2)) #,key="%.5lg_%.5lg_%.gl"%(rt1,rt2,line_energy))

            #fit_results.append([rt1,rt2]+list(p)+[model_1d[m].max(),data_1d[m].max(),model_1d.sum(),model_1d.max(),dwidth(ec,model_1d)])
            fit_results.append([rt1,rt2]+list(p))

            #columns.append(to1d(model_1d))
            columns.append(fitted_model_1d)
            #columns.append(fitted_model_bkg)
            columns.append(data_1d)


        return array(columns),array(fit_results)

    def line_energy_model(self,line_energy,rt1,rt2):
        return self.get_model(line_energy)[2][:,rt1:rt2].sum(1)

    def main(self):
        ec=self.input_linebipars.ec
        de=self.input_linebipars.de

        for line_energy in self.input_linebipars.energies:
            continue
            if line_energy not in [59.3182,511]: continue

            c,f=self.fit_by_rt(lambda *a:[self.line_energy_model(line_energy,*a)])
            
            gaussian_model=self.get_gaussian(line_energy)
            c_g,f_g=self.fit_by_rt(lambda rt1,rt2:gaussian_model)

            print(f.shape,f_g.shape)
        
            np.savetxt("reconstructed_model_1d_%.5lg.txt"%line_energy,np.column_stack(c))
            np.savetxt("model_fit_1d_%.5lg.txt"%line_energy,np.column_stack((f,f_g)))
        
        c_g,f_g=self.fit_by_rt(LineModel)
        print(c_g)
        np.savetxt("reconstructed_model_1d_complex.txt",np.column_stack(c_g))
        np.savetxt("model_fit_1d_complex.txt",f_g)


class Spectrum1DVirtual(da.DataAnalysis):
    input_binned=BinBackgroundSpectrumExtra

    copy_cached_input=False

    def get_h1(self):
        try:
            print("spectrum from ",self.input_binned.h1_energy_rt_16_50.get_path())
            h1_50=np.load(self.input_binned.h1_energy_rt_16_50.open())
            h1_116=np.load(self.input_binned.h1_energy_rt_16_116.open())
        except:
            h1_50=np.load(self.input_binned.h1_energy_lrt50.open())
            h1_116=np.load(self.input_binned.h1_energy_lrt116.open())
        return h1_50[0],h1_116[0]

    def get_h2(self):
        try:
            return self.input_binned.get_h2_energy_pi_300()
        except:
            return pyfits.open(self.input_binned.h2_energy_pi.get_path())[0].data


    def get_exposure(self):
        return self.input_scw.get_telapse()
    
    def get_ebins(self):
        try:
            return np.load(self.input_binned.h1_energy_rt_16_50.open())[1]
        except:
            return np.load(self.input_binned.h1_energy_lrt50.open())[1]

    spectrum_version=None

    def get_spectrum(self):
        if self.spectrum_version is None:
            return self.spectrum
        else:
            return getattr(self,'spectrum_'+self.spectrum_version)

    def get_version(self):
        v=self.get_signature()+"."+self.version
        if self.spectrum_version is not None:
            v+=self.spectrum_version
        return v

    def main(self):
        h1_50,h1_116=self.get_h1()
        ebins=self.get_ebins()

        np.exposure=self.get_exposure()

        counts=h1_50
        
        e1=ebins[:-1]
        e2=ebins[1:]
        rmf=ogip.spec.RMF(e1,e2,e1,e2,np.diag(np.ones_like(e1)))
        fn="response_unitary.fits"
        rmf.write(fn)
        self.rmf=da.DataFile(fn)

        pha=ogip.spec.PHAI(counts,sqrt(counts),np.exposure,response=fn)
        fn="spectrum_lrt50.fits"
        pha.write(fn)
        self.spectrum=da.DataFile(fn)
        self.spectrum_lowrt=da.DataFile(fn)
        
        pha=ogip.spec.PHAI(h1_116,sqrt(h1_116),np.exposure,response=fn)
        fn="spectrum_fullrt.fits"
        pha.write(fn)
        self.spectrum_fullrt=da.DataFile(fn)
        
        h2=self.get_h2()
        if h2 is not None:
            for rt1,rt2 in [(50,80),(80,116),(50,116)]:
                h1=h2[:,rt1:rt2].sum(axis=1).astype(np.float64)
                pha=ogip.spec.PHAI(h1,sqrt(h1),np.exposure,response=fn)
                fn="spectrum_rt_%.5lg_%.5lg.fits"%(rt1,rt2)
                pha.write(fn)
                setattr(self,"spectrum_%.5lg_%.5lg"%(rt1,rt2),da.DataFile(fn))

class Spectrum1D(Spectrum1DVirtual):
    input_scw=ddosa.ScWData

class Spectrum1DP2(Spectrum1D):
    input_binned=BinBackgroundSpectrumExtraP2

class Spectrum1DRev(Spectrum1DVirtual):
    #input_binned=Bipar
    input_binned=BinBackgroundMerged

    version="v1"

    cached=True
    cache=ddosa.DataAnalysis.cache

    copy_cached_input=False

    rtlim=40
    
    def get_ebins(self):
        return np.arange(2049)*0.5
    
    def get_h1(self):
        bipar=self.input_binned.get_h2_pha1_pi()

        h1_lrt=bipar[:,:self.rtlim].sum(axis=1)
        h1_116=bipar[:,:116].sum(axis=1)

        return h1_lrt,h1_116

    def get_exposure(self):
        return 1.
    
class EnergySpectrum1DRev(Spectrum1DRev):
    #input_binned=Bipar
    input_binned=BinBackgroundMerged

    version="v3"
    
    def get_ebins(self):
        return np.arange(2049)*0.5*2.

    def get_h1(self):
        bipar=pyfits.open(self.input_binned.h2_energy_pi.get_path())[0].data

        h1_lrt=bipar[:,:self.rtlim].sum(axis=1)
        h1_116=bipar[:,:116].sum(axis=1)

        return h1_lrt,h1_116


class Spectrum1DRevP2(Spectrum1DRev):
    #input_binned=Bipar
    input_binned=BinBackgroundMergedP2

class EnergySpectrum1DRevP2(EnergySpectrum1DRev):
    #input_binned=Bipar
    input_binned=BinBackgroundMergedP2
    
class BinBackgroundRevP2(ddosa.DataAnalysis):
    input_eventfiles=EventFileListP2
    
    #input_correction=ModelByScWEvoltuion

    cached=True

    copy_cached_input=False

    version="v1"

    highres=1.

    def get_version(self):
        return self.get_signature()+"."+self.version+".hr%.5lg"%self.highres
    
    def correct_energy_p3(self,e,t):
        if not hasattr(self,'input_correction'):
            return e
        print("will correct %.10lg"%np.average(t))
        le_corr=self.input_correction.get_le_model()(t)
        he_corr=self.input_correction.get_he_model()(t)
        print(le_corr,he_corr)

        gain_corr=(he_corr*511-le_corr*59.5)/(511-59.5)
        offset_corr=(le_corr-1)*59.5

        print("gain,offset",np.average(gain_corr),np.average(offset_corr))

        return (e-offset_corr)/gain_corr

    def main(self):
        h2_energy_pi=None
        h2_pha1_pi=None
        h2_pha2_pi=None
        from numpy.random import uniform

        for eventfile in self.input_eventfiles.thelist:

            ijd=None
            if isinstance(eventfile,list):
                scw,eventfile=eventfile
                evts_all=pyfits.open(scw.get_isgri_events())['ISGR-EVTS-ALL'].data
                ijd=evts_all['TIME']

            e_fn=eventfile.events.get_path()
            evts=pyfits.open(e_fn)['ISGR-EVTS-COR'].data
            
            energies=self.correct_energy_p3(evts['ISGRI_ENERGY'],ijd)

            h2=list(np.histogram2d(energies,evts['ISGRI_PI'],bins=(np.logspace(1,np.log10(2048),300*self.highres),np.arange(257))))
            if h2_energy_pi is None:
                h2_energy_pi=h2
            else:
                h2_energy_pi[0]+=h2[0]
            
            h2=list(np.histogram2d(evts['ISGRI_PHA1']/2.+uniform(size=evts.shape[0])-0.5,evts['ISGRI_PI'],bins=(np.logspace(1,np.log10(2048),300*self.highres),np.arange(257))))
            if h2_pha1_pi is None:
                h2_pha1_pi=h2
            else:
                h2_pha1_pi[0]+=h2[0]
            
            h2=list(np.histogram2d(evts['ISGRI_PHA2']+uniform(size=evts.shape[0])-0.5,evts['ISGRI_PI'],bins=(np.logspace(1,np.log10(2048),300*self.highres),np.arange(257))))
            if h2_pha2_pi is None:
                h2_pha2_pi=h2
            else:
                h2_pha2_pi[0]+=h2[0]

            
        
        fn="h2_energy_pi_rev.fits"
        print("bins",h2_energy_pi[1])
        pyfits.HDUList( [pyfits.PrimaryHDU(h2_energy_pi[0]),
                        pyfits.BinTableHDU.from_columns([
                                pyfits.Column(name='EBOUND', format='E', array=h2_energy_pi[1])
                        ])]).writeto(fn,overwrite=True)
        self.h2_energy_pi=da.DataFile(fn)

        np.savetxt("h1_energy.txt",np.column_stack((h2_energy_pi[1][:-1],h2_energy_pi[1][1:],h2_energy_pi[0].sum(axis=1),h2_energy_pi[0][:,:50].sum(axis=1))))
        
        fn="h2_pha1_pi_rev.fits"
        pyfits.HDUList( [pyfits.PrimaryHDU(h2_pha1_pi[0]),
                        pyfits.BinTableHDU.from_columns([
                                pyfits.Column(name='EBOUND', format='E', array=h2_pha1_pi[1])
                        ])]).writeto(fn,overwrite=True)
        self.h2_pha1_pi=da.DataFile(fn)
        
        fn="h2_pha2_pi_rev.fits"
        pyfits.HDUList( [pyfits.PrimaryHDU(h2_pha2_pi[0]),
                        pyfits.BinTableHDU.from_columns([
                                pyfits.Column(name='EBOUND', format='E', array=h2_pha2_pi[1])
                        ])]).writeto(fn,overwrite=True)
        self.h2_pha2_pi=da.DataFile(fn)


class ModelByScWEvoltuion(ddosa.DataAnalysis): pass

class BinBackgroundRev(BinBackgroundRevP2):
    input_eventfiles=EventFileList

class BinBackgroundRevP3(BinBackgroundRevP2):
    input_eventfiles=EventFileListP3

class FitLocalLinesScW(ddosa.DataAnalysis):
    input_spectrum=BinBackgroundSpectrumExtraP2
    #input_spectrum=Spectrum1DP2
    #input_spectrum=EnergySpectrum1DRevP2

    cached=True

    copy_cached_input=False

    version="v3.1"

    mode="energy_pi"

    gain=None
    offset=None

    guess_width=None

    onlyfrt=False
    heline=True
    leline=True

    plot=False

    def get_version(self):
        s=self.get_signature()+"."+self.version
        if self.mode!="energy_pi":
            s+="."+self.mode
        if self.gain is not None:
            s+=".gain%.3lg"%self.gain
        if self.offset is not None:
            s+=".offset%.3lg"%self.offset
        if self.guess_width is not None:
            s+=".gw%.3lg"%self.guess_width
        return s


    def get_h2_energy_pi(self):
        return self.input_spectrum.h2_energy_pi.get_path()
    
    def get_h2_pha1_pi(self):
        return self.input_spectrum.h2_pha1_pi.get_path()
        
    def main(self):

        if self.mode=="energy_pi":
            f=pyfits.open(self.get_h2_energy_pi())
        if self.mode=="pha1_pi":
            f=pyfits.open(self.get_h2_pha1_pi())

        h2=f[0].data
        if len(f)>1:
            ebins=f[1].data['EBOUND']
        else:
            ebins=self.get_ebins()

        self.tag="fullrt"
        self.fit_spectrum(ebins,h2[:,16:116].sum(axis=1))

        if not self.onlyfrt:
            #self.tag="verylowrt"
            #self.fit_spectrum(ebins,h2[:,:16].sum(axis=1))

            self.tag="lowrt"
            self.fit_spectrum(ebins,h2[:,16:50].sum(axis=1))

            self.tag="highrt"
            self.fit_spectrum(ebins,h2[:,50:116].sum(axis=1))

    def fit_spectrum(self,ebins,counts,gain=1,offset=0):
  #      spec=pyfits.open(fn)['SPECTRUM'].data
 #       ebins=pyfits.open(fn)['EBOUNDS'].data

        if self.gain is not None:
            gain=self.gain
        
        if self.offset is not None:
            self.offset=offset

        if counts.shape[0]==2048:
            counts=counts[:-1]

        err=sqrt(counts)
        
        e1=ebins[:-1]
        e2=ebins[1:]
        
        print(counts.shape,e1.shape,e2.shape)
        
        counts/=(e2-e1)
        err/=(e2-e1)


        txt=""
        if self.leline:
            m=np.logical_and(e1>40*gain+offset,e2<70*gain+offset)

            lines=[57.9817*gain+offset,  59.3182*gain+offset]
            fractions=[0.365,0.635]
            offsets=[l-lines[-1] for l in lines]
            guess_width= 5 if self.guess_width is None else self.guess_width
            le_line=self.fit_line(counts[m],err[m],e1[m],e2[m],59.6*gain+offset,guess_width*gain,composite=(offsets,fractions))
            setattr(self,'le_line_'+self.tag,le_line)

            fields=sorted(le_line.keys())
            txt+=" ".join(["%.5lg"%le_line[f] for f in fields])+"\n"
        
        if self.heline:
            m=np.logical_and(e1>300*gain+offset,e2<700*gain+offset)
            he_line=self.fit_line(counts[m],err[m],e1[m],e2[m],511.*gain+offset,20*gain)
            setattr(self,'he_line_'+self.tag,he_line)
            
            fields=sorted(he_line.keys())
            txt+=" ".join(["%.5lg"%he_line[f] for f in fields])+"\n"

        fn="local_lines_%s.txt"%self.tag
        of=open(fn,"w")
        of.write(" ".join(fields)+"\n"+txt)
        of.close()

        setattr(self,'local_lines_'+self.tag+'_fn',da.DataFile(fn))

    def fit_line(self,counts,err,e1,e2,guess_c,guess_w,composite=None):
        
        on_region=np.logical_and(e1>guess_c-guess_w*2,e2<guess_c+guess_w*2)

        ec=(e1*e2)**0.5

        slope, intercept, r_value, p_value, std_err = stats.linregress(np.log(ec[~on_region]),np.log(counts[~on_region])) 
        
        bkg=np.exp(intercept+np.log(ec)*slope)
        

        guess_norm=max(counts-bkg)

        def line_model_basic(x,p):
            N,x0,w,wa=p
            return np.exp(-(x-x0)**2/(w*(1+wa*x/x0))**2/2.)*N

        if composite is None:
            line_model=line_model_basic
        else:
            #offsets,fractions=[[0],[1]]
            offsets,fractions=composite

            def line_model(x,p):
                N,x0,w,wa=p
                return sum([line_model_basic(x,[N*frac,x0+offs,w,wa]) for offs,frac in zip(offsets,fractions)],axis=0)
            
        def line_model_fx0(x,p):
            N,w,wa=p
            return line_model(x,[N,x0_fixed,w,wa])#np.exp(-(x-x0_fixed)**2/(w*(1+wa*x/x0_fixed))**2/2.)*N

        def residual_func(model,p):
            return sum(((counts-bkg-model(ec,p))/err)**2)
        
        import nlopt

        opt = nlopt.opt(nlopt.LN_COBYLA, 4)
        opt.set_lower_bounds([0,guess_c-guess_w,0,-2])
        opt.set_upper_bounds([guess_norm*2,guess_c+guess_w,guess_w*2,2])
        opt.set_min_objective(lambda p,g:residual_func(line_model,p))
        opt.set_xtol_rel(1e-4)
    
        x0=[guess_norm,guess_c,guess_w,0]
        x = opt.optimize(x0)
        optf = opt.last_optimum_value()
        print("optimum at ", x)
        print("optimum value = ", optf)
        print("result code = ", opt.last_optimize_result())

        bestfit=x

        # upper limit
        x0_fixed=bestfit[1]
        while True:
            opt = nlopt.opt(nlopt.LN_COBYLA, 3)
            opt.set_lower_bounds([0,0,-2])
            opt.set_upper_bounds([guess_norm*2,guess_w*2,2])
            opt.set_min_objective(lambda p,g:residual_func(line_model_fx0,p))
            opt.set_xtol_rel(1e-4)
        
            x0_fixed+=bestfit[1]*0.001
            print("trying x0",x0_fixed)
            x_try = opt.optimize([bestfit[0],bestfit[2],bestfit[3]])
            optf_try = opt.last_optimum_value()
            print("optimum at ", x_try)
            print("optimum value = ", optf_try,optf_try-optf)
            print("result code = ", opt.last_optimize_result())

            if optf_try-optf>9.:
                break

            if x0_fixed>ec[-1]:
                break
            
            if np.isnan(x0_fixed):
                break
        x0_upper_limit=x0_fixed

        # lower limit
        x0_fixed=bestfit[1]
        while True:
            opt = nlopt.opt(nlopt.LN_COBYLA, 3)
            opt.set_lower_bounds([0,0,-2])
            opt.set_upper_bounds([guess_norm*2,guess_w*2,2])
            opt.set_min_objective(lambda p,g:residual_func(line_model_fx0,p))
            opt.set_xtol_rel(1e-4)
        
            x0_fixed-=bestfit[1]*0.001
            print("trying x0",x0_fixed)
            x_try = opt.optimize([bestfit[0],bestfit[2],bestfit[3]])
            optf_try = opt.last_optimum_value()
            print("optimum at ", x_try)
            print("optimum value = ", optf_try,optf_try-optf)
            print("result code = ", opt.last_optimize_result())

            if optf_try-optf>9.:
                break
            if x0_fixed<ec[0]:
                break
            if np.isnan(x0_fixed):
                break
        x0_lower_limit=x0_fixed

        centroid=sum(ec*line_model(ec,x))/sum(line_model(ec,x))
        width=sqrt(sum(ec*ec*line_model(ec,x))/sum(line_model(ec,x))-centroid**2)
        
        if self.plot:
            plot.p.clf()
            plot.p.errorbar(ec,counts,err)
            plot.p.errorbar(ec[~on_region],counts[~on_region],err[~on_region])
            plot.p.plot(ec,bkg)
            plot.p.plot(ec,line_model(ec,x)+bkg)
            l=line_model(ec,x)
            plot.p.plot(ec,l/l.max()*(l+bkg)[on_region].max())
            plot.p.title("x0: %.5lg , %.5lg - %.5lg (%.5lg +- %.5lg), %s"%(bestfit[1],x0_lower_limit,x0_upper_limit,centroid,(x0_upper_limit-x0_lower_limit)/2.,self.tag))
            pngfn="line_%.5lg_%s.png"%(guess_c,self.tag)
            plot.plot(pngfn)


        print("peak, centroid, width:",bestfit[1],centroid,width,(bestfit[1]-guess_c)/guess_c,(centroid-guess_c)/guess_c)
                
        return dict(centroid=centroid,width=width,
                    bestfit_N=bestfit[0],
                    bestfit_x0=bestfit[1],
                    bestfit_w=bestfit[2],
                    bestfit_wa=bestfit[3],
                    x0_lower_limit=x0_lower_limit,
                    x0_upper_limit=x0_upper_limit)

class FitLocalLinesScWP0(FitLocalLinesScW):
    input_spectrum=BinBackgroundSpectrumExtra

        
class FitLocalLinesRevP2(FitLocalLinesScW):
    input_spectrum=BinBackgroundRevP2
    #input_spectrum=Spectrum1DP2
    #input_spectrum=EnergySpectrum1DRevP2

class FitLocalLinesRevCorrected(FitLocalLinesScW):
    input_spectrum=CorrectBipar

class FitLocalLinesRevCorrectedP4(FitLocalLinesScW):
    input_spectrum=CorrectBiparP4
    #input_spectrum=Spectrum1DP2
    #input_spectrum=EnergySpectrum1DRevP2

class FitLocalLinesRevP0(FitLocalLinesScW):
    input_spectrum=BinBackgroundRev
    #input_spectrum=Spectrum1DP2
    #input_spectrum=EnergySpectrum1DRevP2

class FitLocalLinesMergedP0(FitLocalLinesScW):
    input_spectrum=BinBackgroundMerged
    #input_spectrum=Spectrum1DP2
    #input_spectrum=EnergySpectrum1DRevP2

    def get_ebins(self):
        return np.linspace(0,2048,2048)

    def get_h2_energy_pi(self):
        print(dir(self.input_spectrum))
        return self.input_spectrum.h2_energy_pi.get_path()


class FitLocalLinesRevP2(FitLocalLinesScW):
    input_spectrum=BinBackgroundRevP2
    #input_spectrum=Spectrum1DP2
    #input_spectrum=EnergySpectrum1DRevP2

class FitLocalLinesRevP3(FitLocalLinesScW):
    input_spectrum=BinBackgroundRevP3

class Estimate3DModel(Fit3DModel):
    only_estimation=True
    save_corrected_est=True

class FitLocalLinesCorrected(FitLocalLinesScW):
    input_spectrum=Fit3DModel
    
    guess_width=3.
    
    def get_h2_energy_pi(self):
        return self.input_spectrum.data_corrected_frommodel.get_path()

class FitLocalLinesCorrectedEstimation(FitLocalLinesScW):
    input_spectrum=Estimate3DModel

    guess_width=3.
    
    def get_h2_energy_pi(self):
        return self.input_spectrum.data_corrected__final.get_path()
    
    #def get_h2_pha1_pi(self):
    #    return self.input_spectrum.h2_pha1_pi.get_path()

#class Spectrum1DCorrectedEstimation(FitLocalLinesScW):
#    input_spectrum=Estimate3DModel
class Spectrum1DCorrected(Spectrum1DVirtual):
    input_spectrum=CorrectBipar

    guess_width=3.
    
    def get_h2_energy_pi(self):
        return pyfits.open(self.input_spectrum.h2_energy_pi.get_path())
    
    copy_cached_input=False

    def get_h2(self):
        return self.get_h2_energy_pi()[0].data

    def get_exposure(self):
        return 1
    
    def get_ebins(self):
        return self.get_h2_energy_pi()[1].data['EBOUND']

    spectrum_version="16_116"

    def get_spectrum(self):
        if self.spectrum_version is None:
            return self.spectrum
        else:
            return getattr(self,'spectrum_'+self.spectrum_version)

    def get_version(self):
        v=self.get_signature()+"."+self.version
        if self.spectrum_version is not None:
            v+=self.spectrum_version
        return v

    def main(self):
        ebins=self.get_ebins()
        np.exposure=self.get_exposure()
        
        e1=ebins[:-1]
        e2=ebins[1:]
        rmf=ogip.spec.RMF(e1,e2,e1,e2,np.diag(np.ones_like(e1)))
        rmf_fn="response_unitary.fits"
        rmf.write(rmf_fn)
        self.rmf=da.DataFile(rmf_fn)

        h2=self.get_h2()
        for rt1,rt2 in [(16,50),(16,116),(50,80),(80,116),(50,116)]:
            h1=h2[:,rt1:rt2].sum(axis=1).astype(np.float64)
            pha=ogip.spec.PHAI(h1,sqrt(h1),np.exposure,response=rmf_fn)
            fn="spectrum_rt_%.5lg_%.5lg.fits"%(rt1,rt2)
            pha.write(fn)
            setattr(self,"spectrum_%.5lg_%.5lg"%(rt1,rt2),da.DataFile(fn))


class BestGuessEnergy(da.DataAnalysis):
    input_rev=Revolution

    def main(self):
        ht_avail=fit_ng.list_ht()

        revnum=int(self.input_rev.input_revid.handle)

        revs=[[float(re.search("complete_(\d+?)_pha.xcm",k.split("/")[-1]).groups()[0]),k] for k in ht_avail if re.search("complete_(\d+?)_pha.xcm",k.split("/")[-1])]

        best=sorted(revs,key=lambda x:abs(x[0]-revnum))[0]
    
        print(best)

        self.xcm_filename=best[1]

        self.parameters=dict(offset=(0,0.01,-20,-20,20,20),gain=(1,0.001,0.5,0.5,1.5,1.5))

class BestGuessRev(da.DataAnalysis):
    input_peaks=FindPeaks
    input_rev=Revolution

    freeze_gain2=False

    def get_version(self):
        if self.freeze_gain2:
            return self.get_signature()+"."+self.version+".frg2"
        else:
            return self.get_signature()+"."+self.version
    
    def main(self):
        ht_avail=fit_ng.list_ht()

        revnum=int(self.input_rev.input_revid.handle)

        revs=[[float(re.search("complete_(\d+?)_pha.xcm",k.split("/")[-1]).groups()[0]),k] for k in ht_avail if re.search("complete_(\d+?)_pha.xcm",k.split("/")[-1])]

        best=sorted(revs,key=lambda x:abs(x[0]-revnum))[0]
    
        print(best)

        self.xcm_filename=best[1]

        self.parameters=dict(offset=self.input_peaks.offset/2.,gain=self.input_peaks.gain/2.)
        if self.freeze_gain2:
            self.parameters['gain2']=(0,-1,0,0,100,100)
        print("best guess parameters:",self.parameters)


class LUT2Files(da.DataAnalysis):
    pass

class LineSamples(da.DataAnalysis):
    #input_fit=Fit3DModel

    def main(self):
        sample_energies=[30,40,60,100,150,250,400,511]

        s=None
        for energy in sample_energies:
            d=self.input_biparmodel.bipar_model.generate_bipar(self.input_biparmodel.bipar_model.detector(),energy)[0]
            if s is None:
                s=d
            else:
                s+=d
        pyfits.PrimaryHDU(s).writeto("sample_energies.fits",overwrite=True)


class Response2D(ddosa.DataAnalysis):
    input_lut2=GenerateLUT2
    input_biparmodel=LocateBiparModel

    copy_cached_input=False

    version="v4"

    cached=True

    def main(self):
        print("will np.load r3d..")
        
        lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())[0].data

        r3d=self.input_biparmodel.bipar_model.response3d(None,grouping=0.01)
        r3d.np.loadfrom(self.input_lut2.response_3d.get_path())
        
        energies=np.linspace(0,1024.,1024)
        #energies=np.linspace(1,1500**0.5,300)**2

        keys=[255,116,90,100,50,(16,116),(16,50),125,135,145,155,165]
        r2s, files = self.input_biparmodel.bipar_model.response_response(lut2,r3d,energies,savefits=True,limitrt=keys)

        self.r1ds=[]
        for a,b in zip(keys,files):
            if isinstance(a,tuple):
                n="r1d_lrt%lg-%lg"%a
            else:
                n="r1d_lrt%lg"%a
            setattr(self,n,da.DataFile(b))
            print("storing",n,b,getattr(self,n))
            self.r1ds.append([a,n])

 #       pyfits.PrimaryHDU(result).writeto("synth_bipar.fits",overwrite=True)
  #      self.synth_bipar=da.DataFile("synth_bipar.fits")

class InspectLUT2(da.DataAnalysis):
    input_lut2=FinalizeLUT2 

    def main(self):
        lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())[0].data

        print(lut2)

        #ebins=np.linspace(0,1000,2001)
        ebins=np.logspace(1,np.log10(2048),4800)
        inverted=[]
        for i in range(256):
            h1=np.histogram(lut2[:,i],ebins)[0]
            inverted.append(h1)

        pyfits.PrimaryHDU(np.transpose(inverted)).writeto("lut2_inverted.fits",overwrite=True)

class Response(da.DataAnalysis):
    input_r2d=Response2D
    input_ic=ddosa.IBIS_ICRoot

    copy_cached_input=False

    use_flat_arf=True

    version="v2200"

    def main(self):
        arf_hdu=pyfits.open("/workdir/savchenk/projects/ibismm/test1/arf.fits")['ISGR-ARF.-RSP']
        #arf_hdu=pyfits.open("/np.savedir/astrohe/savchenk/mc_ibis_mm_me/t1/arf.fits")['ISGR-ARF.-RSP']
        #arf_hdu=pyfits.open("/np.savedir/astrohe/savchenk/mc_ibis_mm_me/t1/run/arf.fits")['ISGR-ARF.-RSP']
        #arf_hdu=pyfits.open(self.input_ic.ibisicroot+"/mod/isgr_effi_mod_0011.fits")['ISGR-ARF.-RSP']
        arf_mc=arf_hdu.data
        arf_e1,arf_e2=arf_mc['ENERG_LO'],arf_mc['ENERG_HI']
        arf_0=copy(arf_mc['SPECRESP'])

    ##
        if self.use_flat_arf:
            arf_0=np.ones(2200)
            arf_e1=0.5*np.arange(2200)
            arf_e2=0.5*np.arange(1,2201)
    ##
        arf_e0=(arf_e2+arf_e1)/2.

        np.savetxt("arf.txt",arf_0)


        for key,r2dattr in self.input_r2d.r1ds:
            f=pyfits.open(getattr(self.input_r2d,r2dattr).get_path())
            r2d=f[0].data
            arf_l2_e=f[1].data['ENERGY']

            r2d[:10,:]=0
            r2d[:,:10]=0
            r2d[np.isnan(r2d)]=0

            print(r2d.shape)

            np.savetxt("r2_0.txt",r2d.sum(axis=0))
            np.savetxt("r2_1.txt",r2d.sum(axis=1))

            arf_l2=r2d.sum(axis=1)
            arf_l2_interpolated=i1d(arf_l2_e,arf_l2,bounds_error=False)(arf_e0)
        
            arf_l2_interpolated[np.isnan(arf_l2_interpolated)]=0

            np.savetxt("arf%s1.txt"%key,arf_0)
            np.savetxt("arf%s2.txt"%key,arf_0)

            if self.use_flat_arf:
                ogip.spec.ARF(arf_e1,arf_e2,arf_l2_interpolated).write("arf_%s_lut2.fits"%key)
            else:
                arf_hdu.data['SPECRESP']=arf_0*arf_l2_interpolated
                arf_hdu.writeto("arf_%s.fits"%key,overwrite=True)

                arf_hdu.data['SPECRESP']=arf_l2_interpolated
                arf_hdu.writeto("arf_%s_lut2.fits"%key,overwrite=True)
        
            np.savetxt("arfs_%s.txt"%key,np.column_stack((arf_0,arf_l2_interpolated)))

class RMFFile(da.DataAnalysis):
    def main(self):
        self.binrmf=da.DataFile("/workdir/savchenk/projects/ibismm/test1/rmf.fits")


class EfficiencyUpdate(da.DataAnalysis):
    input_r2d=Response2D
    input_rmf=RMFFile
    #input_rmf=ddosa.SpectraBins

    copy_cached_input=False

    def main(self):
        rmf_f=pyfits.open(self.input_rmf.binrmf.get_path())
        rmf_matrix_orig=copy(np.row_stack(array(rmf_f['MATRIX'].data['MATRIX'])))

        specbins_hdu=rmf_f[1].data
        #specbins_hdu=pyfits.open(self.input_rmf.binrmf)['ISGR-EBDS-MOD'].data
        e1=specbins_hdu['E_MIN']
        e2=specbins_hdu['E_MAX']
        spec_energies=(e1+e2)/2.

        from scipy.interpolate import interp1d as i1d
        from scipy.ndimage.filters import gaussian_filter as g2f
            
        total_r2d=g2f(pyfits.open(self.input_r2d.r1d_lrt255.get_path())[0].data,3)
        #total_r2d=g2f(pyfits.open(self.input_r2d.r1d_lrt255.get_path())[0].data,3)
        #norms=[1/sum(total_r2d[:,i]) for i in range(total_r2d.shape[1])]

        for key,r2dattr in self.input_r2d.r1ds:
            f=pyfits.open(getattr(self.input_r2d,r2dattr).get_path())
            r2d=g2f(f[0].data,3)


            #r2d[:40,:]=0
            #r2d[:,:40]=0
            r2d[np.isnan(r2d)]=0

            print(r2d.shape)

            np.savetxt("r2_0.txt",r2d.sum(axis=0))
            np.savetxt("r2_1.txt",r2d.sum(axis=1))

            energies=np.arange(2048)*0.5

            effi_l2=r2d.sum(axis=0)#*norms
            

            effi_l2_interpolated=i1d(energies,effi_l2,bounds_error=False)(spec_energies)
            norm=effi_l2_interpolated[spec_energies>40.][0]
            print("norm",norm)
            effi_l2_interpolated/=norm
            effi_l2_interpolated[spec_energies<50]=1
        
            effi_l2_interpolated[np.isnan(effi_l2_interpolated)]=0

            effi_hdu = pyfits.BinTableHDU.from_columns([
                     pyfits.Column(name='ENERGY', format='E', array=spec_energies),
                     pyfits.Column(name='EFFICIENCY', format='E', array=effi_l2_interpolated)])

            fn="effi_%s.fits"%key
            pyfits.HDUList([pyfits.PrimaryHDU(),effi_hdu]).writeto(fn,overwrite=True)

            np.savetxt("effi_%s.txt"%key,np.column_stack((spec_energies,effi_l2_interpolated)))

            setattr(self,"effi_%s"%key,da.DataFile(fn))


            for i in range(rmf_matrix_orig.shape[0]):
                n=rmf_matrix_orig[i]*effi_l2_interpolated
                n[np.isnan(n)]=0
                n[np.isinf(n)]=0
                rmf_f['MATRIX'].data['MATRIX'][i]=n
                #print n
            rmf_f.writeto("rmf_effi_%s.fits"%key,overwrite=True)

            arf=np.row_stack(array(rmf_f['MATRIX'].data['MATRIX'])).sum(axis=1)
            np.savetxt("arf_%s.txt"%key,arf)
            #np.savetxt("arf_%s.txt"%key,np.column_stack((arf)))

            #/Integral/data/ic_collection/ic_tree-20130107/ic/ibis/rsp/isgr_arf_rsp_0031.fits 

            #arf_hdu.data['SPECRESP']=arf_l2_interpolated
            #arf_hdu.writeto("arf_%s_lut2.fits"%key,overwrite=True)


class SynthBiparFrom1D(da.DataAnalysis):
    input_1d=EnergySpectrum1DRevP2
    input_lut2=GenerateLUT2

    copy_cached_input=False

    cached=True

    def main(self):

        print("will synth")
        #lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())

        r3d=self.input_biparmodel.bipar_model.response3d(None,grouping=0.01)
        r3d.np.loadfrom(self.input_lut2.response_3d.get_path())

        spec=pyfits.open(self.input_1d.spectrum.get_path())[1].data['COUNTS']
        ebe=pyfits.open(self.input_1d.rmf.get_path())['EBOUNDS'].data
        e1,e2=ebe['E_MIN'],ebe['E_MAX']
        ec=(e1+e2)*0.5
        de=(e2-e1)

        spec_f=interp1d(ec,spec/(e2-e1),fill_value=0,bounds_error=False)

        result=None
        for en in np.linspace(1,1500**0.5,200)**2:
            amp=spec_f(en)
            print(en,amp)
            r=r3d.get_energy(en,disable_ai=False)
            if r is None: continue
            print(r.sum())
            if result is None:
                result=np.zeros_like(r)
            m_amp=r[:,:50].sum() # which?
            mt_amp=r.sum() # which?
            print("lrt model amp",m_amp,"full",mt_amp)
            if m_amp<mt_amp*1e-5: continue
            if amp==0: continue
            print("adding")
            r=r*amp/m_amp*mt_amp

            #if abs(en-511)<2: r=r*10

            r[np.isnan(r)]=0
            r[np.isinf(r)]=0
            result+=r

        pyfits.PrimaryHDU(result).writeto("synth_bipar.fits",overwrite=True)

        self.synth_bipar=da.DataFile("synth_bipar.fits")
        

class Fit1DSpectrumRev(fit_ng.Fit1Dglobal):
    input_bestguess=BestGuessRev
    input_spectrum=Spectrum1DRev

    cached=True
    cache=ddosa.DataAnalysis.cache

class Fit1DSpectrumRevNNL(fit_ng.Fit1Dglobal):
    input_bestguess=BestGuessRev(use_freeze_gain2=True)
    input_spectrum=Spectrum1DRev

    cached=True
    cache=ddosa.DataAnalysis.cache


class BestGuessByScw(da.DataAnalysis):
    input_scw=ddosa.ScWData
    
    def main(self):
        revnum=float(self.input_scw.input_scwid.handle[:4])
        print(revnum)
        
        ht_avail=fit_ng.list_ht()
        #ht_avail=glob.glob("/home/savchenk/fit1d/xspec_model/humantouch/*xcm")
    
        revs=[[float(re.search("complete_(\d+?).xcm",k.split("/")[-1]).groups()[0]),k] for k in ht_avail if re.search("complete_(\d+?).xcm",k.split("/")[-1])]

        best=sorted(revs,key=lambda x:abs(x[0]-revnum))[0]
    
        print(best)

        self.xcm_filename=best[1]

class Fit1DSpectrum(fit_ng.Fit1Dglobal):
    input_bestguess=BestGuessByScw
    input_spectrum=Spectrum1DP2

    cached=True
    cache=ddosa.DataAnalysis.cache
    
    #override_parameters={'offset':(0,-1,-1,-1,1,1)}

class Fit1DSpectrumRevLines(fit_ng.Fit1DglobalLines):
    input_bestguess=BestGuessRev #(use_freeze_gain2=True)
    input_spectrum=Spectrum1DRev

    cached=True
    cache=ddosa.DataAnalysis.cache

class Fit1DSpectrumLines(fit_ng.Fit1DglobalLines):
    input_bestguess=BestGuessByScw
    input_spectrum=Spectrum1DP2

    cached=True
    cache=ddosa.DataAnalysis.cache
    
    #override_parameters={'offset':(0,-1,-1,-1,1,1)}

class Fit1DSpectrumRevNNLLines(fit_ng.Fit1DglobalLines):
    input_bestguess=BestGuessRev(use_freeze_gain2=True)
    input_spectrum=Spectrum1DRev

    cached=True
    cache=ddosa.DataAnalysis.cache

class Fit1DSpectrumRevEnergy(fit_ng.Fit1Dglobal):
    input_bestguess=BestGuessEnergy
    input_spectrum=EnergySpectrum1DRevP2

    cached=True
    cache=ddosa.DataAnalysis.cache

class Fit1DSpectrumRevEnergyLines(fit_ng.Fit1DglobalLines):
    input_bestguess=BestGuessEnergy
    input_spectrum=EnergySpectrum1DRevP2

    cached=True
    cache=ddosa.DataAnalysis.cache
    
    #override_parameters={'offset':(0,-1,-1,-1,1,1)}

class Fit1DSpectrumCorrected(fit_ng.Fit1DglobalLines):
    input_bestguess=BestGuessEnergy
    input_spectrum=Spectrum1DCorrected

    gain=1
    offset=0

    cached=True
    cache=ddosa.DataAnalysis.cache
    

class FineEnergyCorrection(ddosa.DataAnalysis):
    input_correction=ModelByScWEvoltuion
    input_p2events=ISGRIEventsScW
    input_scw=ddosa.ScWData

    copy_cached_input=False
    
    cached=True
   # cached=False
    cache=cache_local
    
    read_caches=[cache_local.__class__]
    write_caches=[cache_local.__class__]

    version="v2"


    def main(self):
        evts_all=pyfits.open(self.input_scw.get_isgri_events())['ISGR-EVTS-ALL'].data
        t=evts_all['TIME']

        print("will correct %.10lg"%np.average(t))
        le_corr=self.input_correction.get_le_model()(t)
        he_corr=self.input_correction.get_he_model()(t)
        print(le_corr,he_corr)

        gain_corr=(511-59.5)/(he_corr-le_corr)
        offset_corr=59.5-le_corr*gain_corr

        print("gain,offset",np.average(gain_corr),np.average(offset_corr))
        print("gain,offset will do",le_corr*gain_corr+offset_corr)
        print("gain,offset will do",he_corr*gain_corr+offset_corr)


        f=pyfits.open(self.input_p2events.events.get_cached_path())
        e=f['ISGR-EVTS-COR']

        newenergy=e.data['ISGRI_ENERGY']*gain_corr+offset_corr

        print("np.average offset",np.average(newenergy-e.data['ISGRI_ENERGY']))

        #e.insert_column(name='ISGRI_ENERGY_P2',data=e.data['ISGRI_ENERGY'])
        e.data['ISGRI_ENERGY']=newenergy

        #e_log=np.logspace(0,3,1000)
        #np.savetxt("correction_p2_p3.txt",np.column_stack((e_log,correct_energy(e_log))))

        f.writeto("isgri_energy_scw_p3.fits",overwrite=True)
        self.events=da.DataFile("isgri_energy_scw_p3.fits")


class BinBackgroundSpectrumP3(BinBackgroundSpectrum):
    input_events=FineEnergyCorrection
    tag="P3"

class BinBackgroundSpectrumP4(BinBackgroundSpectrum):
    input_events=ISGRIEventsScWP4
    tag="P3"

class Spectrum1DP3(Spectrum1D):
    input_binned=BinBackgroundSpectrumP3

class BinBackgroundSpectrumExtraP3(BinBackgroundSpectrumP3):
    save_extra=True

class FitLocalLinesScWP3(FitLocalLinesScW):
    input_spectrum=BinBackgroundSpectrumExtraP3

class ISGRIEventsFinal(da.DataAnalysis):
    events=ISGRIEventsScW
    #events=FineEnergyCorrection

    allow_alias=False
    run_for_hashe=True

    def main(self):
        return self.events

class ISGRIEventsScWP4(ISGRIEventsScW):
    input_evttag = ibis_isgr_evts_tag_scw_P4

class BinEventsVirtual(ddosa.BinEventsVirtual):
    input_events=ISGRIEventsFinal
    input_lut2=FinalizeLUT2

    #input_ltdata=ltdata.LTStat #??

    #version="hotfix"
    version="flexlt1"
    biaslt=0

    forcelt=None

    ltpick=None
    ltfrac=None

    #ii_shadow_build_binary=os.environ['COMMON_INTEGRAL_SOFTDIR']+"/ii_shadow_build/ii_shadow_build_osa102le/ii_shadow_build"
    ii_shadow_build_binary="ii_shadow_build"
    #ii_shadow_build_binary=os.environ['COMMON_INTEGRAL_SOFTDIR']+"/ii_shadow_build/ii_shadow_build_lt/ii_shadow_build"

    def get_version(self):
        v=ddosa.BinEventsVirtual.get_version(self)

        if self.forcelt is not None:
            v+=".forcelt_%.5lg_%.5lg"%self.forcelt
        else:
            if self.biaslt!=0:
                v+=".biaslt%.5lg"%self.biaslt

        if self.ltpick is not None:
            v+=".ltpick%i"%self.ltpick
        elif self.ltfrac is not None:
            v+=".ltfrac%.5lg"%self.ltfrac
        return v

    def pre_process(self):
        lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())[0].data
        
        ref_ch=[30,40,50]
        ref_e=np.nanmean(lut2[ref_ch,16:20],1)-self.biaslt

        print("ref_ch, ref_e",ref_ch,ref_e)

        import scipy.stats

        gain,offset=scipy.stats.linregress(ref_e,ref_ch)[:2]

        print("computed LE offset, gain:",offset,gain)

        if self.forcelt is not None:
            offset,gain=self.forcelt
            print("forced LE offset, gain:",offset,gain)
    
        open("gains.txt","w").write("%.5lg %.5lg\n"%(gain,offset))

    def post_process(self):
        if hasattr(self,'input_ltdata'):
            iy,iz=np.meshgrid(np.arange(128),np.arange(128))
            y=iy+(iy/32)*2
            z=iz+(iz/64)*2

            ltmap_pixels=np.loadtxt(self.input_ltdata.ltmap.get_path())
            iltmap=np.histogram2d(ltmap_pixels[:,1],ltmap_pixels[:,0],weights=ltmap_pixels[:,3],bins=(np.arange(129)-0.5,np.arange(129)-0.5))[0]

            ltmap=np.zeros((134,130))
            ltmap[y,z]=iltmap[iy,iz]
            lt_lim=np.percentile(ltmap,90)
            lth=np.histogram(ltmap,np.linspace(ltmap[ltmap>0].min(),lt_lim,30))
            ltbins=array(list(zip(lth[1][:-1],lth[1][1:])))
            print("lt bins:",ltbins)
            pyfits.PrimaryHDU(ltmap).writeto("ltmap.fits",overwrite=True)

            s=np.argsort(lth[0])[::-1]
            print(list(zip(ltbins[s[:10]],lth[0][s[:10]])))
            
            if self.ltfrac is not None:
                assert(self.ltfrac!=0)
                assert(abs(self.ltfrac)<=1)
                if self.ltfrac>0:
                    mask=iltmap<=np.percentile(ltmap.flatten(),self.ltfrac*100)
                else:
                    mask=iltmap>=np.percentile(ltmap.flatten(),100+self.ltfrac*100)

                print("ltfraction",iltmap[mask].min(),iltmap[mask].max(),iltmap[mask].max(),sum(mask))

                f=pyfits.open(self.shadow_efficiency.get_path())
                for i in range(len(f[2:])):
                    f[2+i].data[~mask]=0

                fn="shadow_efficiency_ltpick.fits"
                f.writeto(fn,overwrite=True)
                self.shadow_efficiency=da.DataFile(fn)
            elif self.ltpick is not None:
                ltbin=ltbins[s[self.ltpick]]
                print("ltbin",ltbin,"with",lth[0][s[self.ltpick]])
                mask=(iltmap>=ltbin[0]) & (iltmap<ltbin[1])

                f=pyfits.open(self.shadow_efficiency.get_path())
                for i in range(len(f[2:])):
                    f[2+i].data[~mask]=0

                fn="shadow_efficiency_ltpick.fits"
                f.writeto(fn,overwrite=True)
                self.shadow_efficiency=da.DataFile(fn)


class BinEventsImage(BinEventsVirtual):
    target_level="BIN_I"
    input_bins=ImageBins

class BinEventsSpectra(BinEventsVirtual):
    target_level="BIN_S"
    input_bins=SpectraBins

class ii_spectra_correct(ddosa.DataAnalysis): # many ways
    input_spectra=ddosa.ii_spectra_extract
    input_effi=EfficiencyUpdate

    def main(self):
        f=pyfits.open(self.input_spectra.spectrum.get_path())
        effi=pyfits.open(self.input_effi.effi_100.get_path())[1].data['EFFICIENCY']

        for i in f[2:]:
            i.data['RATE']=i.data['RATE']/effi
            i.data['STAT_ERR']=i.data['STAT_ERR']/effi
            i.data['SYS_ERR']=i.data['SYS_ERR']/effi

        f.writeto("isgri_spectrum_corrected.fits",overwrite=True)


class OnAxisCorr(ddosa.DataAnalysis):
    input_maps=ddosa.BinMapsSpectra

    def main(self):
        self.onaxis_corr=[[(f.header['E_MIN']+f.header['E_MIN'])/2.,f.data[64,64]] for f in pyfits.open(self.input_maps.corr.get_path())[2:]]
        #self.onaxis_corr=[f.data.max() for f in pyfits.open(self.input_maps.corr.get_path())[2:]]

        np.savetxt("onaxis_corr.txt",self.onaxis_corr)


class BiSpectrum(ddosa.DataAnalysis):
    input_spectrum=ii_spectra_extract
    input_events=ISGRIEvents
    input_events_final=ISGRIEventsScW
    input_scw=ddosa.ScWData
    input_rmf=ddosa.SpectraBins
    input_onaxiscorr=OnAxisCorr
    input_effi=ddosa.BinEventsSpectra

    cached=True
    copy_cached_input=False

    version="v6"

    def main(self):
        f_pifs=pyfits.open(self.input_spectrum.pifs.get_path())

        print(f_pifs)

        pif=None
        for e_pif in f_pifs[2:]:
            print(e_pif.header['NAME'].strip())
            if e_pif.header['NAME'].strip()=='Crab': pif=e_pif.data

        print(pif)

        if pif is None:
            self.empty_results="no pif"
            return

        #  reshape pif

        def reshape(dete):
            dete=dete[:,:]
            dete[:,64:128]=dete[:,66:130]
            dete[:,128:]=0

            dete[32:64,:]=dete[34:66,:]
            dete[64:96,:]=dete[68:100,:]
            dete[96:128,:]=dete[102:134,:]
            dete[128:,:]=0
            return dete

        pif=reshape(pif)

        # / reshape pif 


        pyfits.PrimaryHDU(pif).writeto("pif_128.fits",overwrite=True)

        events=pyfits.open(self.input_events.events.get_path())['ISGR-EVTS-COR'].data
        events_final=pyfits.open(self.input_events_final.events.get_path())['ISGR-EVTS-COR'].data
        events_all=pyfits.open(self.input_scw.scwpath+"/isgri_events.fits.gz")['ISGR-EVTS-ALL'].data

        print(events,pif)

        pif_values=pif[events_all['ISGRI_Z'],events_all['ISGRI_Y']]

        print(pif_values.shape)

        s_mask=pif_values>0.75
        b_mask=pif_values<0.25

        area_on=sum(pif>0.75)
        area_off=sum(pif<0.25)
        
        # efficeincy
        effi=[np.average(f.data) for f in pyfits.open(self.input_effi.shadow_efficiency.get_path())[2:]]

        print(effi)

        # np.average?..
        # efficeincy
        
        s_yz=np.histogram2d(events_all[s_mask]['ISGRI_Z'],events_all[s_mask]['ISGRI_Y'],bins=(np.arange(134),np.arange(134)))[0]/s_mask.shape[0]
        pyfits.PrimaryHDU(s_yz).writeto("s_yz.fits",overwrite=True)
        b_yz=np.histogram2d(events_all[b_mask]['ISGRI_Z'],events_all[b_mask]['ISGRI_Y'],bins=(np.arange(134),np.arange(134)))[0]/b_mask.shape[0]
        pyfits.PrimaryHDU(b_yz).writeto("b_yz.fits",overwrite=True)
        pyfits.PrimaryHDU(s_yz-b_yz).writeto("sb_yz.fits",overwrite=True)
        all_yz=np.histogram2d(events_all['ISGRI_Z'],events_all['ISGRI_Y'],bins=(np.arange(134),np.arange(134)))[0]
        pyfits.PrimaryHDU(all_yz).writeto("all_yz.fits",overwrite=True)

        s_pha1_pi=np.histogram2d(events[s_mask]['ISGRI_PHA1'],events[s_mask]['ISGRI_PI'],bins=(np.arange(2048),np.arange(256)))[0]
        b_pha1_pi=np.histogram2d(events[b_mask]['ISGRI_PHA1'],events[b_mask]['ISGRI_PI'],bins=(np.arange(2048),np.arange(256)))[0]/area_off*area_on

        pyfits.PrimaryHDU(s_pha1_pi).writeto("s_pha1_pi.fits",overwrite=True)
        pyfits.PrimaryHDU(s_pha1_pi-b_pha1_pi).writeto("sb_pha1_pi.fits",overwrite=True)
        pyfits.PrimaryHDU(b_pha1_pi).writeto("b_pha1_pi.fits",overwrite=True)
        self.s_pha1_pi=da.DataFile("s_pha1_pi.fits")
        self.sb_pha1_pi=da.DataFile("sb_pha1_pi.fits")
        self.b_pha1_pi=da.DataFile("b_pha1_pi.fits")

        specbins_hdu=pyfits.open(self.input_rmf.binrmf)['ISGR-EBDS-MOD'].data
        e1=specbins_hdu['E_MIN']
        e2=specbins_hdu['E_MAX']
        self.ebins=np.concatenate((e1,[e2[-1]]))
        binw=(self.ebins[1:]-self.ebins[:-1])
        
        onaxis_corr=self.input_onaxiscorr.onaxis_corr
        self.onaxis_corr=self.input_onaxiscorr.onaxis_corr

        pixel_area=1 #0.4**2        

        s_energy_pi=np.histogram2d(events_final[s_mask]['ISGRI_ENERGY'],events_final[s_mask]['ISGRI_PI'],bins=(self.ebins,np.arange(256)))[0]/np.outer(onaxis_corr,np.ones(255))/np.outer(effi,np.ones(255))/pixel_area/np.outer(binw,np.ones(255))
        b_energy_pi=np.histogram2d(events_final[b_mask]['ISGRI_ENERGY'],events_final[b_mask]['ISGRI_PI'],bins=(self.ebins,np.arange(256)))[0]/area_off*area_on/np.outer(onaxis_corr,np.ones(255))/np.outer(effi,np.ones(255))/pixel_area/np.outer(binw,np.ones(255))
        
        pyfits.PrimaryHDU(s_energy_pi).writeto("s_energy_pi.fits",overwrite=True)
        pyfits.PrimaryHDU(s_energy_pi-b_energy_pi).writeto("sb_energy_pi.fits",overwrite=True)
        pyfits.PrimaryHDU(b_energy_pi).writeto("b_energy_pi.fits",overwrite=True)
        self.s_energy_pi=da.DataFile("s_energy_pi.fits")
        self.sb_energy_pi=da.DataFile("sb_energy_pi.fits")
        self.b_energy_pi=da.DataFile("b_energy_pi.fits")

        
        #[[f.header['E_MAX'],f.data.max()] for f in open(self.input_maps.corr)[2:]]

        self.np.exposure=pyfits.open(self.input_spectrum.spectrum.get_path())[2].header['EXPOSURE']

class BiSpectrumProcessingSummary(ddosa.DataAnalysis):
    run_for_hashe=True

    def main(self):
        mf=BiSpectrum(assume=ddosa.ScWData(input_scwid="any",use_abstract=True)) # arbitrary choice of scw, should be the same: assumption of course
        ahash=mf.process(output_required=False,run_if_haveto=False)[0]
        print("generalized hash:",ahash)
        rh=dataanalysis.shhash(ahash)
        print("reduced hash",rh)
        return [dataanalysis.DataHandle('processing_definition:'+rh[:8])]

class BiSpectrumList(ddosa.DataAnalysis):
    input_scwlist=ddosa.RevScWList
    input_will_use_promise=BiSpectrumProcessingSummary
    copy_cached_input=False
    allow_alias=True

    def main(self):
        self.thelist=[]
        for s in self.input_scwlist.scwlistdata:
            a=BiSpectrum(assume=s)
            print(a,a.assumptions)
            self.thelist.append(a)

class BiSpectrumMerged(ddosa.DataAnalysis):
    input_list=BiSpectrumList

    copy_cached_input=False
    cached=True
    

    version="v2"

    def main(self):
        print(self.input_list)

        sb_energy_pi=None
        s_energy_pi=None
        sb_pha1_pi=None
        s_pha1_pi=None
        b_pha1_pi=None
        np.exposure=0
        onaxis_corr=None
        ebins=None
        for bs in self.input_list.thelist:
            if hasattr(bs,'empty_results'): continue

            if onaxis_corr is None:
                onaxis_corr=bs.onaxis_corr

            if ebins is None:
                ebins=bs.ebins
    
            np.exposure+=bs.np.exposure
            e_p=pyfits.open(bs.sb_energy_pi.get_path())[0].data

            if sb_energy_pi is None:
                sb_energy_pi=e_p
            else:
                sb_energy_pi+=e_p
            
            e_p=pyfits.open(bs.s_energy_pi.get_path())[0].data
            if s_energy_pi is None:
                s_energy_pi=e_p
            else:
                s_energy_pi+=e_p


            p_p=pyfits.open(bs.sb_pha1_pi.get_path())[0].data
            if sb_pha1_pi is None:
                sb_pha1_pi=p_p
            else:
                sb_pha1_pi+=p_p

            
            p_p=pyfits.open(bs.s_pha1_pi.get_path())[0].data
            if s_pha1_pi is None:
                s_pha1_pi=p_p
            else:
                s_pha1_pi+=p_p
            
            p_p=pyfits.open(bs.b_pha1_pi.get_path())[0].data
            if b_pha1_pi is None:
                b_pha1_pi=p_p
            else:
                b_pha1_pi+=p_p


        binw=(ebins[1:]-ebins[:-1])

        pyfits.PrimaryHDU(sb_energy_pi).writeto("sb_energy_pi_merged.fits",overwrite=True)
        self.sb_energy_pi_merged=da.DataFile("sb_energy_pi_merged.fits")
        
        pyfits.PrimaryHDU(sb_pha1_pi).writeto("sb_pha1_pi_merged.fits",overwrite=True)
        self.sb_pha1_pi_merged=da.DataFile("sb_pha1_pi_merged.fits")
        
        pyfits.PrimaryHDU(b_pha1_pi).writeto("b_pha1_pi_merged.fits",overwrite=True)
        self.b_pha1_pi_merged=da.DataFile("b_pha1_pi_merged.fits")
        
        pyfits.PrimaryHDU(s_pha1_pi).writeto("s_pha1_pi_merged.fits",overwrite=True)
        self.s_pha1_pi_merged=da.DataFile("s_pha1_pi_merged.fits")

        a=pyfits.open("/Integral/data/ic_collection/ic_tree-20130108/ic/ibis/mod/isgr_effi_mod_0011.fits")['ISGR-ARF.-RSP'].data
        me1,me2=a['ENERG_LO'],a['ENERG_HI']

       # ebins=np.logspace(1,3,100)
       # a,b=np.meshgrid(ebins[:-1],me1)
       # np.diagonal=np.exp(-(a-b)**2/2)

       # ogip.spec.RMF(ebins[:-1],ebins[1:],me1,me2,np.diagonal).write("response_np.diag_log100.fits")
        
        #self.response_np.diag=da.DataFile("response_np.diag_log100.fits")

        s1d=sb_energy_pi[:,16:116].sum(axis=1)
        s1dt=s_energy_pi[:,16:116].sum(axis=1)

        self.np.exposure=np.exposure

        ogip.spec.PHAI(s1d*binw,sqrt(s1dt)*binw,np.exposure).write("energy_spectrum.fits")
        ogip.spec.PHAI(s1d*binw*onaxis_corr,sqrt(s1dt)*binw*onaxis_corr,np.exposure).write("energy_spectrum_noonaxis.fits")
        
        self.energy_spectrum=da.DataFile("energy_spectrum.fits")

    def get_h2_pha1_pi(self):
        print("opening",self.b_pha1_pi.path)
        return  pyfits.open(self.b_pha1_pi.get_path())[0].data

class ReconstructBipar(da.DataAnalysis):
    input_bipar=BiSpectrumMerged
    #input_bipar=BinBackgroundMerged
    input_lut2=GenerateLUT2
    input_rmf=ddosa.SpectraBins

    copy_cached_input=False
    test_input=False
        
    def main(self):
        specbins_hdu=pyfits.open(self.input_rmf.binrmf)['ISGR-EBDS-MOD'].data
        e1=specbins_hdu['E_MIN']
        e2=specbins_hdu['E_MAX']
        new_energies=(e1+e2)/2.
        binw=(e2-e1)

        pha1_pi=pyfits.open(self.input_bipar.sb_pha1_pi_merged.get_path())[0].data
        self.reconstruct(pha1_pi,"source",new_energies,binw)
        
        new_energies=np.linspace(0,1024,1024)
        binw=new_energies*0+0.5
        pha1_pi=pyfits.open(self.input_bipar.b_pha1_pi_merged.get_path())[0].data
        self.reconstruct(pha1_pi,"background",new_energies,binw)

    def reconstruct(self,pha1_pi,key,new_energies,binw):
        #pha1_pi=pyfits.open(self.input_bipar.b_pha1_pi_merged.get_path())[0].data
       # pha1_pi=self.input_bipar.get_h2_pha1_pi()

        try:
            np.exposure=self.input_bipar.np.exposure
        except:
            np.exposure=100000

 #       pha,rt=np.mgrid(np.arange(2048),np.arange(256))

        lut2=pyfits.open(self.input_lut2.lut2_1d.get_path())[0].data
        
        from scipy.interpolate import interp1d as i1d

        lut2[:20,:]=0

        energy_pi=[]
        for rt in np.arange(255):

            spec_row=pha1_pi[:,rt]
            energy_row=lut2[:,rt]

            energy_row=energy_row - 2.*((rt-20.)/(80-20))**3 # * (np.exp(-(energy_row-60)**2/50))

            energy_row=energy_row[:-1]

            print(spec_row.shape,energy_row.shape)

            energy_pi.append(i1d(energy_row,spec_row,bounds_error=False,fill_value=0)(new_energies),bounds_error=False)
            np.savetxt("spec_energy_row_%i.txt"%rt,np.column_stack((spec_row,energy_row))) #,new_energies,energy_pi[-1][:])))
            np.savetxt("spec_newenergy_row_%i.txt"%rt,np.column_stack((new_energies,energy_pi[-1][:])))

        energy_pi=np.transpose(array(energy_pi))


        pyfits.PrimaryHDU(energy_pi).writeto("energy_pi_%s.fits"%key,overwrite=True)
        s1d=energy_pi[:,16:116].sum(axis=1)
        s1dt=energy_pi[:,16:116].sum(axis=1) # not same

        ogip.spec.PHAI(s1d*binw,sqrt(s1dt)*binw,np.exposure/10).write("energy_spectrum_recon_%s.fits"%key)



class FitLocalLinesList(da.DataAnalysis):
    input_scwlist=ddosa.RevScWList

    copy_cached_input=False
    allow_alias=True

    def main(self):
        self.thelist=[]
        l=self.input_scwlist.scwlistdata
        for s in l:
            a=FitLocalLinesScW(assume=s,input_scw=s)
            print(a,a.assumptions)
            self.thelist.append((s,a))

class FitLocalLinesListP3(da.DataAnalysis):
    input_scwlist=ddosa.RevScWList

    copy_cached_input=False
    allow_alias=True

    def main(self):
        self.thelist=[]
        l=self.input_scwlist.scwlistdata
        for s in l:
            a=FitLocalLinesScWP3(assume=s,input_scw=s)
            print(a,a.assumptions)
            self.thelist.append((s,a))


class FitLocalLinesByScW(ddosa.DataAnalysis):
    input_fitbyscw=FitLocalLinesList

    cached=True
    copy_cached_input=False

    def main(self):
        for scw,fit in self.input_fitbyscw.thelist:
            print(fit.le_line_lowrt,fit.le_line_highrt,fit.le_line_fullrt)
            print(fit.he_line_lowrt,fit.he_line_highrt,fit.he_line_fullrt)

        t,dt=list(zip(*[scw.get_t() for scw,fit in self.input_fitbyscw.thelist]))
        c_le_lrt,c_le_lrt_err=list(zip(*[[fit.le_line_lowrt['centroid'],(fit.le_line_lowrt['x0_upper_limit']-fit.le_line_lowrt['x0_lower_limit'])/2.] for scw,fit in self.input_fitbyscw.thelist]))
        c_le_hrt,c_le_hrt_err=list(zip(*[[fit.le_line_highrt['centroid'],(fit.le_line_highrt['x0_upper_limit']-fit.le_line_highrt['x0_lower_limit'])/2.] for scw,fit in self.input_fitbyscw.thelist]))
        c_le_frt,c_le_frt_err=list(zip(*[[fit.le_line_fullrt['centroid'],(fit.le_line_fullrt['x0_upper_limit']-fit.le_line_fullrt['x0_lower_limit'])/2.] for scw,fit in self.input_fitbyscw.thelist]))
        
        c_he_lrt,c_he_lrt_err=list(zip(*[[fit.he_line_lowrt['centroid'],(fit.he_line_lowrt['x0_upper_limit']-fit.he_line_lowrt['x0_lower_limit'])/2.] for scw,fit in self.input_fitbyscw.thelist]))
        c_he_hrt,c_he_hrt_err=list(zip(*[[fit.he_line_highrt['centroid'],(fit.he_line_highrt['x0_upper_limit']-fit.he_line_highrt['x0_lower_limit'])/2.] for scw,fit in self.input_fitbyscw.thelist]))
        c_he_frt,c_he_frt_err=list(zip(*[[fit.he_line_fullrt['centroid'],(fit.he_line_fullrt['x0_upper_limit']-fit.he_line_fullrt['x0_lower_limit'])/2.] for scw,fit in self.input_fitbyscw.thelist]))

        self.results=np.column_stack((t,dt,c_le_lrt,c_le_lrt_err,c_le_hrt,c_le_hrt_err,c_le_frt,c_le_frt_err,c_he_lrt,c_he_lrt_err,c_he_hrt,c_he_hrt_err,c_he_frt,c_he_frt_err))
        np.savetxt("lines.txt",self.results)
        self.results_file=da.DataFile("lines.txt")

class FitLocalLinesByScWP3(FitLocalLinesByScW):
    input_fitbyscw=FitLocalLinesListP3

class ModelByScWEvoltuion(ddosa.DataAnalysis):
    input_linesbyscw=FitLocalLinesByScW
     
    cached=True

    def main(self):
        results=self.input_linesbyscw.results

        t=results[:,0]
        
        le_lrt=results[:,2]
        le_e_lrt=results[:,3]
        le_hrt=results[:,4]
        le_e_hrt=results[:,5]
        le_frt=results[:,6]
        le_e_frt=results[:,7]

        he_lrt=results[:,8]
        he_e_lrt=results[:,9]
        he_hrt=results[:,10]
        he_e_hrt=results[:,11]
        he_frt=results[:,12]
        he_e_frt=results[:,13]
        


        self.le_model=self.fit_evolution(t,le_frt,le_e_frt)
        self.he_model=self.fit_evolution(t,he_frt,he_e_frt)
        self.le_model_lrt=self.fit_evolution(t,le_lrt,le_e_lrt,"_lrt")
        self.he_model_lrt=self.fit_evolution(t,he_lrt,he_e_lrt,"_lrt")
        self.le_model_hrt=self.fit_evolution(t,le_hrt,le_e_hrt,"_hrt")
        self.he_model_hrt=self.fit_evolution(t,he_hrt,he_e_hrt,"_hrt")


    def get_le_model(self):
        (t0,en0),p=self.le_model
        #return lambda x:( (1+p[0]+(x-t0)*p[1]+(x-t0)**2*p[2])*e ) #!!!
        return lambda x:( (1+p[0]+(x-t0)*p[1]+(x-t0)**2*p[2])*en0 ) #!!!
    
    def get_he_model(self):
        (t0,en0),p=self.he_model
        #return lambda x:( (1+p[0]+(x-t0)*p[1]+(x-t0)**2*p[2]) ) #!!!
        return lambda x:( (1+p[0]+(x-t0)*p[1]+(x-t0)**2*p[2])*en0 ) #!!!

    def fit_evolution(self,t,en,en_e,key=""):

        t0=np.average(t)
        en0=sum(en/en_e**2)/sum(1/en_e**2)
    
        print("reference t0,en0",t0,en0)

        model=lambda x,p:( (1+p[0]+(x-t0)*p[1]+(x-t0)**2*p[2])*en0 ) #!!!
        residuals=lambda p,g:np.average((model(t,p)-en)**2/en_e**2)

        import nlopt
        opt = nlopt.opt(nlopt.LN_COBYLA, 3)
        opt.set_lower_bounds([-10,-10,-10])
        opt.set_upper_bounds([10,10,10])
        opt.set_min_objective(residuals)
        opt.set_xtol_rel(1e-4)
    
        p = opt.optimize([0,0,0])
        optf = opt.last_optimum_value()
        print("optimum at ", p)
        print("optimum value = ", optf)
        print("result code = ", opt.last_optimize_result())

        plot.p.clf()
        plot.p.errorbar(t,en,en_e)
        plot.p.plot(t,model(t,p))
        plot.plot("line_%.10lg_%.5lg%s.png"%(t0,en0,key))

        return (t0,en0),p

class BinBackgroundMergedP3(BinBackgroundMergedP2):
    input_list=BinBackgroundListP2
    input_correction=ModelByScWEvoltuion
    input_scwlist=ddosa.RevScWList
    
    #output_tag="P3"
    

# carf carf=lambda x:(1+0.15/(1+np.exp(-((x-60)/5))))

class StudySpectrumRevP2(ddosa.DataAnalysis):
    input_spectrum=BinBackgroundRevP2
    #input_spectrum=Spectrum1DP2
    #input_spectrum=EnergySpectrum1DRevP2


    copy_cached_input=False

    version="v1"

    def get_h2_energy_pi(self):
        return self.input_spectrum.h2_energy_pi.get_path()

    def main(self):
        f=pyfits.open(self.get_h2_energy_pi())

        ebins=f[1].data['EBOUND']
        h2=f[0].data

        self.tag="fullrt"
        self.plot_spectrum(ebins,h2[:,16:116].sum(axis=1))

        self.tag="lowrt"
        self.plot_spectrum(ebins,h2[:,16:50].sum(axis=1))

        self.tag="highrt"
        self.plot_spectrum(ebins,h2[:,50:116].sum(axis=1))

    def plot_spectrum(self,ebins,h1):
        plot.p.clf()
        plot.p.errorbar(ebins[:-1],h1,h1**0.5,lw=0)
        np.savetxt("energy_1d_%s.txt"%self.tag,np.column_stack((ebins[:-1],h1,h1**0.5)))
        plot.p.loglog()
        plot.plot("energy_%s.png"%self.tag)


