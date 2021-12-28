import numpy as np
from netCDF4 import Dataset
from scipy import stats, special, integrate
import cmath
import bhmie
import pickle

#DEPENDENCIES
#Modules: standard python packages except bhmie
#Download bhmie.py from https://github.com/lo-co/atm-py/blob/master/atmPy/for_removal/mie/bhmie.py
#Files: P3IceLookupTable.nc and m_ice_tables

#INPUT
#required variables for all schemes are 
#physics: name of a microphysics scheme, either M2005,
#         M2005_HMmod, SAM1MOM, THOM, or P3
#qc (cloud droplet mass mixing ratio) [g/kg]
#qi (cloud ice mass mixing ratio) [g/kg]
#qs (snow mass mixing ratio) [g/kg]
#qr (rain mass mixing ratio) [g/kg]
#qg (graupel mass mixing ratio) [g/kg]
#p (pressure) [mb]
#t (temperature) [K]
#rho (density) [kg/m3]
#nc (droplet number concentration) [cm-3]

#additional required variables for M2005 and M2005_MOD
#ni (cloud ice number concentration) [cm-3]
#nr (rain number concentration) [cm-3]
#ns (snow number concentration) [cm-3]
#ng (graupel number concentration) [cm-3]

#additional required variables for THOM
#ni (cloud ice number concentration) [cm-3]
#nr (rain number concentration) [cm-3]

#additional required variables for P3
#qir (rime ice volume mixing ratio) [g/kg]
#qib (rime ice mass mixing ratio) [cm-3/kg]

#if P3 scheme only outputs the total 
#frozen mass mixing ratio (qi+qs+qg),
#then input that for qi and input equal-size arrays 
#filled with zeros for qs and qg

#OUTPUT

#dBZ_nonatt (unatteuated total reflectivity)
#z_effc (unattenuated reflectivity factor for droplets)
#z_effi (unattenuated reflectivity factor for cloud ice)
#z_effr (unattenuated reflectivity factor for rain)
#z_effs (unattenuated reflectivity factor for snow)
#z_effg (unattenuated reflectivity factor for graupel)

#reflectivity factors equal zero where there is too little
#hydrometeor mass
#If there is no cloud or precipitation, dBZ_nonatt=-inf

#z_effc and z_effi are always zero for SAM1MOM
#z_effs and z_effg are always zero for P3

def radar_simulator(physics,qc,qi,qs,qr,qg,p,t,rho,nc,ni=np.nan,nr=np.nan,\
                    ns=np.nan,ng=np.nan,qir=np.nan,qib=np.nan):

  #Radar frequency
  freq=94.4

  #Create size bins following QUICKBEAM
  nd=85
  dmin=.1
  dmax=1e4
  D=np.zeros(nd)
  D[0]=dmin
  delt=np.exp((np.log(dmax)-np.log(dmin))/(nd-1))
  for i in range(nd-1):
    D[i+1]=D[i]*delt
  D=D*1e-6
  D_mid=D[0:-1]+np.diff(D)/2.
  dD=np.diff(D)
  dlogD=np.diff(np.log10(D))

  #Call microphysics codes to get particle size distributions (PSDs)
  if ((physics=='M2005')or(physics=='M2005_HMmod')):
    [Dc,Di,Dr,Ds,Dg,dDc,dDi,dDr,dDs,dDg,\
     Nc,Ni,Nr,Ns,Ng]=\
     get_psd_m2005(D,D_mid,dD,nc,ni,ns,nr,ng,qc,qi,qs,qr,qg,rho)
  if physics=='SAM1MOM':
    [Dc,Di,Dr,Ds,Dg,dDc,dDi,dDr,dDs,dDg,\
     Nc,Ni,Nr,Ns,Ng]=\
     get_psd_sam1mom(D,D_mid,dD,qs,qr,qg,rho)
  if physics=='THOM':
    [Dc,Di,Dr,Ds,Dg,dDc,dDi,dDr,dDs,dDg,\
     Nc,Ni,Nr,Ns,Ng]=\
     get_psd_thom(D,D_mid,dD,nc,ni,nr,qc,qi,qs,qr,qg,rho,t)
  if (physics=='P3'):
    [Dc,Di,Dr,dDc,dDi,dDr,Nc,Ni,Nr]=\
     get_psd_p3(D,D_mid,dD,nc,ni,nr,qc,qi+qs+qg,qr,qir,qib,rho)

  #Compute index of refraction for ice
  [n_r,n_i]=compute_m_ice(freq,t)
  m_ice=complex(n_r,-n_i)

  #Compute index of refraction for liquid water
  [n_r,n_i]=compute_m_wat(freq,t)
  m_wat=complex(n_r,-n_i)

  #if there is at least a small amount of hydrometeor mass present,
  #compute the unattentuated reflectivity factor (z_eff*) 
  #for that hydrometeor

  #also return reflectivity factor for Rayleight scattering (z_ray*)
  #and attentuation coefficient (kr*)
  #these variables are not currently used but are returned so that 
  #attenuation can be added later

  #SAM1MOM microphysics only assumes size distributions for 
  #precipitation (rain, snow and graupel) so reflectivity is
  #only estimated for those hydrometeors

  if ((qi>1e-10)&(physics!='SAM1MOM')):
    [kri,z_effi,z_rayi]=\
    zeff(Ni,Di,dDi,t,m_ice,ice=True)
  else:
    kri=z_effi=z_rayi=0.0 
 
  if ((qs>1e-10)&(physics!='P3')):
    [krs,z_effs,z_rays]=\
    zeff(Ns,Ds,dDs,t,m_ice,ice=True)
  else:
    krs=z_effs=z_rays=0.0 

  if qr>1e-10:
    [krr,z_effr,z_rayr]=\
    zeff(Nr,Dr,dDr,t,m_wat)
  else:
    krr=z_effr=z_rayr=0.0 

  if ((qg>1e-10)&(physics!='P3')):
    [krg,z_effg,z_rayg]=\
    zeff(Ng,Dg,dDg,t,m_ice,ice=True)
  else:
    krg=z_effg=z_rayg=0.0 

  if ((qc>1e-10)&(physics!='SAM1MOM')):
    [krc,z_effc,z_rayc]=\
    zeff(Nc,Dc,dDc,t,m_wat)
  else:
    krc=z_effc=z_rayc=0.0 

  #Compute unattenuated reflectivity from the unattenuated 
  #reflectivity factors for each hydrometeor
  dBZ_nonatt=10*np.log10(z_effi+z_effs+\
          z_effr+z_effg+z_effc)

  return dBZ_nonatt,z_effc,z_effi,z_effr,z_effs,z_effg

####################################################
#Use SAM1MOM microphysics to compute PSDs###
####################################################
def get_psd_sam1mom(D,D_mid,dD,qs,qr,qg,rho):

  rhor=1000. #kg/m3
  rhos=100. #kg/m3
  rhog=400. #kg/m3

  N0r=8.e6
  N0s=3.e6
  N0g=4.e6

  def nrsg_dist(D_mid,qh,rhoh,rho,N0):
    return N0*np.exp(-(np.pi*rhoh*N0/(qh*rho*1e-3))**(1/4)*D_mid)

  distr_array=nrsg_dist(D_mid,qr,rhor,rho,N0r)
  dists_array=nrsg_dist(D_mid,qs,rhos,rho,N0s)
  distg_array=nrsg_dist(D_mid,qg,rhog,rho,N0g)

  Deqr=D*(rhor/917)**(1/3.)
  Deqs=D*(rhos/917)**(1/3.)
  Deqg=D*(rhog/917)**(1/3.)
  Deqr_mid=Deqr[0:-1]+np.diff(Deqr)/2.
  Deqs_mid=Deqs[0:-1]+np.diff(Deqs)/2.
  Deqg_mid=Deqg[0:-1]+np.diff(Deqg)/2.
  dDeqr=np.diff(Deqr)
  dDeqs=np.diff(Deqs)
  dDeqg=np.diff(Deqg)

  return np.nan*D_mid,np.nan*D_mid,Deqr_mid,Deqs_mid,Deqg_mid,\
         np.nan*dD,np.nan*dD,dDeqr,dDeqs,dDeqg,\
         np.nan*D_mid,np.nan*D_mid,distr_array,\
         dists_array,distg_array 

####################################################
#Use Morrison microphysics to compute PSDs###
####################################################
def get_psd_m2005(D,D_mid,dD,nc,ni,ns,nr,ng,qc,qi,qs,qr,qg,rho):

  rhoi=500 #kg/m3
  rhow=997 #kg/m3
  rhos=100 #kg/m3
  rhog=900 #kg/m3

  pgam=0.0005714*nc+0.2714
  pgam=1./(pgam**2.)-1.
  pgam=np.clip(pgam,2.,10.)

  DI=3.
  CI=rhoi*np.pi/6.
  DCS=125.e-6
  DS=3.
  DG=3.
  CG=rhog*np.pi/6.
  CS=rhos*np.pi/6.
  CONS12=special.gamma(1.+DI)*CI
  CONS26=np.pi*rhow/6.
  CONS1=special.gamma(1.+DS)*CS
  CONS2=special.gamma(1.+DG)*CG

  lamc=(CONS26*nc*1e9*special.gamma(pgam+4.)/\
       (qc*rho*special.gamma(pgam+1.)))**(1/3.)
  lammin=np.min((pgam+1.)/60e-6)
  lammax=np.min((pgam+1.)/1e-6)
  lamc=np.clip(lamc,lammin,lammax)
  n0c=(nc*lamc**(pgam+1.0))/special.gamma(pgam+1.0)

  lami=(CONS12*ni*1e9/\
       (qi*rho))**(1/DI)
  lammin=1./(2.*DCS+100e-6)
  lammax=1.e6
  lami=np.clip(lami,lammin,lammax)

  lamr=(np.pi*rhow*nr*1e9/(qr*rho))**(1/3.)
  lammin=1./28.e-4
  lammax=5.e4
  lamr=np.clip(lamr,lammin,lammax)

  lams=(CONS1*ns*1e9/(qs*rho))**(1./DS)
  lammin=5.e2
  lammax=1.e5
  lams=np.clip(lams,lammin,lammax)

  lamg=(CONS2*ng*1e9/(qg*rho))**(1./DG)
  lammin=5.e-2
  lammax=5.e6
  lamg=np.clip(lamg,lammin,lammax)

  n0i=ni*lami
  n0r=nr*lamr
  n0s=ns*lams
  n0g=ng*lamg

  def nc_dist(x,n0c,pgamc,lamc):
    return n0c*x**pgamc*np.exp(-lamc*x)

  def nirsg_dist(x,n0,lam):
    return n0*np.exp(-lam*x)

  distc_array=nc_dist(D_mid,n0c,pgam,lamc)*1e6
  disti_array=nirsg_dist(D_mid,n0i,lami)*1e6
  distr_array=nirsg_dist(D_mid,n0r,lamr)*1e6
  dists_array=nirsg_dist(D_mid,n0s,lams)*1e6
  distg_array=nirsg_dist(D_mid,n0g,lamg)*1e6

  Deqi=D*(rhoi/917)**(1/3.)
  Deqs=D*(rhos/917)**(1/3.)
  Deqg=D*(rhog/917)**(1/3.)
  Deqi_mid=Deqi[0:-1]+np.diff(Deqi)/2.
  Deqs_mid=Deqs[0:-1]+np.diff(Deqs)/2.
  Deqg_mid=Deqg[0:-1]+np.diff(Deqg)/2.
  dDeqi=np.diff(Deqi)
  dDeqs=np.diff(Deqs)
  dDeqg=np.diff(Deqg)

  return D_mid,Deqi_mid,D_mid,Deqs_mid,Deqg_mid,\
         dD,dDeqi,dD,dDeqs,dDeqg,\
         distc_array,disti_array,distr_array,\
         dists_array,distg_array 

####################################################
#Use P3 microphysics to compute PSDs###
####################################################
def get_psd_p3(D,D_mid,dD,nc,ni,nr,qc,qi,qr,qir,qib,rho):

  rhow=997 #kg/m3

  pgam=0.0005714*nc+0.2714
  pgam=1./(pgam**2.)-1.
  pgam=np.clip(pgam,2.,15.)

  cons1=np.pi*rhow/6.
  lamc = (cons1*1e9*nc*(pgam+3.)*(pgam+2.)*(pgam+1.)/qc*rho)**(1/3.)

  mu_r=1.0
  lamr = (cons1*1e9*nr*(mu_r+3.)*(mu_r+2.)*(mu_r+1.)/qr*rho)**(1/3.)

  n0c=(nc*lamc**(pgam+1.0))/special.gamma(pgam+1.0)
  n0r=(nr*lamr**(mu_r+1.0))/special.gamma(mu_r+1.0)

  def ncr_dist(x,n0c,pgamc,lamc):
    return n0c*x**pgamc*np.exp(-lamc*x)

  data=Dataset('P3IceLookupTable.nc')
  qi_steps=data.variables['qitot'][:]
  rhorime_steps=data.variables['rhorime'][:]
  lambdas=data.variables['lambda_ice'][:]
  Frime=data.variables['Frime'][:]

  Fr=qir/qi #[unitless]
  crp=qir*1e3/qib #[kg m-3]
  qnorm=qi*rho*1e-3/(ni*1e6) #[kg]
  qnear=np.argmin(abs(qi_steps-qnorm))
  rhonear=np.argmin(abs(rhorime_steps-crp))
  Fnear=np.argmin(abs(Fr-Frime))
  table_lam=lambdas[qnear,Fnear,rhonear]

  def ni_dist(x,ni,qi,qir,qib,rho,table_lam):

    Fr=qir/qi

    if Fr>0.0:
      crp=qir/qib
    else:
      crp=0.0

    ds=1.9
    cs=0.0121
    dg=3
    dcrit = (np.pi/(6.*cs)*900.)**(1./(ds-3.))

    for i in range(10000):
      cgp=crp
      if Fr>1.0:
        Fr=1.0
        csr=0.0
      if Fr==0:
        dcrits = .1
        dcritr = dcrits
        csr = cs
        dsr = ds
      elif Fr<1.0:
        dcrits = (cs/cgp)**(1./(dg-ds))
        dcritr = ((1.+Fr/(1.-Fr))*cs/cgp)**(1./(dg-ds))
        csr = cs*(1.+Fr/(1.-Fr))
        dsr = ds
        rhodep = 1./(dcritr-dcrits)*6.*cs/(np.pi*(ds-2.))*\
                 (dcritr**(ds-2.)-dcrits**(ds-2.))
        cgpold = cgp
        cgp = crp*Fr+rhodep*(1.-Fr)*np.pi/6.
        if (abs((cgp-cgpold)/cgp)<0.01):
          break
      elif Fr==1.0:
        dcrits = (cs/cgp)**(1./(dg-ds))
        dcritr = 1e5
        csr = cgp
        dsr = dg

    def rho_i(D,dcrit,dcrits,dcritr,cs,ds,cgp,dg,csr,dsr):
      rho_i=np.zeros(len(D)) 
      rho_i[np.where(D<dcrit)]=917. #solid ice
      index=np.where((D>=dcrit)&(D<dcrits))
      rho_i[index]=cs*D[index]**ds/(np.pi*D[index]**3/6)#unrimed large ice
      index=np.where((D>=dcrits)&(D<dcritr))
      rho_i[index]=cgp*D[index]**dg/(np.pi*D[index]**3/6)#partially rimed ice
      index=np.where(D>=dcritr)
      rho_i[index]=csr*D[index]**dsr/(np.pi*D[index]**3/6)#partially rimed ice
      return rho_i

    def mass(D_mid,ni,lami,mu_i,c,d,rho):
      n0i=(ni*lami**(mu_i+1.0))/special.gamma(mu_i+1.0)
      return n0i*D_mid**mu_i*np.exp(-lami*D_mid)*c*D_mid**d*1e9/rho

    def number(D_mid,ni,lami):
      mu_i=np.clip(0.076*((1/lami)*1e4)**0.8-2.,0.0,6.0)
      n0i=(ni*lami**(mu_i+1.0))/special.gamma(mu_i+1.0)
      return n0i*D_mid**mu_i*np.exp(-lami*D_mid)

    def total_mass(x,qi,dcrit,dcrits,dcritr,cs,ds,cgp,dg,csr,dsr,ni,rho):
      mu_i=np.clip(0.076*((1/x)*1e4)**0.8-2.,0.0,6.0)

      return np.abs(integrate.quad(mass,0.0,dcrit,\
             args=(ni,x,mu_i,np.pi*900/6,3,rho))[0]+\
             integrate.quad(mass,dcrit,dcrits,\
             args=(ni,x,mu_i,cs,ds,rho))[0]+\
             integrate.quad(mass,dcrits,dcritr,\
             args=(ni,x,mu_i,cgp,dg,rho))[0]+\
             integrate.quad(mass,dcritr,1e5,\
             args=(ni,x,mu_i,csr,dsr,rho))[0]-qi)

    return number(D_mid,ni,table_lam),\
           rho_i(D,dcrit,dcrits,dcritr,cs,ds,cgp,dg,csr,dsr)

  distc_array=ncr_dist(D_mid,n0c,pgam,lamc)*1e6
  distr_array=ncr_dist(D_mid,n0r,mu_r,lamr)*1e6
  if ((qi>0.0)&(ni>0.0)):
    disti_array,rhoi=ni_dist(D_mid,ni,qi,qir,qib,rho,table_lam)
    disti_array=disti_array*1e6
    Deqi=D*(rhoi/917)**(1/3.)
    Deqi_mid=Deqi[0:-1]+np.diff(Deqi)/2.
    dDeqi=np.diff(Deqi)
  else:
    disti_array=np.nan*D_mid
    Deqi_mid=np.nan*D_mid
    dDeqi=np.nan*dD

  return D_mid,Deqi_mid,D_mid,dD,dDeqi,dD,\
         distc_array,disti_array,distr_array


####################################################
#Use Thompson microphysics to compute PSDs###
####################################################
def get_psd_thom(D,D_mid,dD,nc,ni,nr,qc,qi,qs,qr,qg,rho,T):

  nr=nr*1e-6*rho
  ni=ni*1e-6*rho
  T=T-273.15

  rhoi=890 #kg/m3
  rhow=1000 #kg/m3
  rhog=500 #kg/m3

  mu_c = 12.
  am_r = np.pi*rhow/6.0
  bm_r = 3.0
  cce1 = mu_c + 1.
  cce2 = bm_r + mu_c + 1.
  cce3 = bm_r + mu_c + 4.
  ccg1 = special.gamma(cce1)
  ccg2 = special.gamma(cce2)
  ccg3 = special.gamma(cce3)
  ocg1 = 1./ccg1
  ocg2 = 1./ccg2
  obmr = 1./bm_r
  lamc = (nc*1e9*am_r*ccg2*ocg1/(qc*rho))**obmr
  N0_c = nc*1e-3*ocg1*lamc**cce1

  mu_i=0.0
  bv_i = 1.0
  am_i = np.pi*rhoi/6.0
  bm_i=3.0
  cie1= mu_i + 1.
  cie2 = bm_i + mu_i + 1.
  cig1 = special.gamma(cie1)
  cig2 = special.gamma(cie2)
  oig1 = 1./cig1
  oig2 = 1./cig2
  obmi = 1./bm_i
  lami = (am_i*cig2*oig1*ni*1e9/(qi*rho))**obmi
  N0_i = ni*1e-3*oig1*lami**cie1

  mu_r=0.0
  am_r = np.pi*rhow/6.0
  bm_r = 3.0
  cre2 = mu_r + 1.
  cre3 = bm_r + mu_r + 1.
  crg2 = special.gamma(cre2)
  crg3 = special.gamma(cre3)
  org2=1./crg2
  org3=1./crg3
  obmr=1./bm_r
  lamr=(am_r*crg3*org2*nr*1e9/(qr*rho))**obmr
  N0_r=nr*1e-3*org2*lamr**cre2

  bm_g = 3.0
  mu_g=0
  obmg=1/bm_g
  cge1 = bm_g + 1.
  cge2 = mu_g + 1.
  cge3 = bm_g + mu_g + 1.
  obmg=1/bm_g
  cgg1 = special.gamma(cge1)
  cgg2 = special.gamma(cge2)
  cgg3 = special.gamma(cge3)
  oge1 = 1./cge1
  ogg1 = 1./cgg1
  ogg2 = 1./cgg2
  ogg3 = 1./cgg3
  am_g = np.pi*rhog/6.0
  mvd_r = (3.0+mu_r+0.672)/lamr
  xslw1=.01
  if (T<-2.5)&(mvd_r>100.e-6):
    xslw1=4.01 + np.log10(mvd_r)
  ygra1=4.31+np.log10(5.E-5)
  if qg*rho>5e-5:
    4.31+np.log10(qg*rho)
  zans1 = 3.1+(100./(300.*xslw1*ygra1/(10./xslw1+1.+0.25*ygra1)+\
          30.+10.*ygra1))
  N0_exp = 10.**(zans1)
  lam_exp = (N0_exp*am_g*cgg1/(qg*rho))**oge1
  lamg =lam_exp*(cgg3*ogg2*ogg1)**obmg
  N0_g = 1e-12*N0_exp/(cgg2*lam_exp)*lamg**cge2

  Kap0 = 490.6
  Kap1 = 17.46
  mu_s = 0.6357
  Lam0 = 20.78
  Lam1 = 3.29
  am_s = 0.069
  bm_s = 2.0
  oams = 1./am_s
  obms = 1./bm_s
  ocms = oams**obms
  smob = qs*rho*1e-3*oams
  a0=13.6
  b0=-.0361
  c0=.807
  a2=13.6-7.76*2+.479*4
  b2=-.0361+.0151*2+.00149*4
  c2=.807+.00581*2+.0457*4
  a3=13.6-7.76*3+.479*9
  b3=-.0361+.0151*3+.00149*9
  c3=.807+.00581*3+.0457*9
  logM2=(np.log(smob)-a2-b2*T)/c2
  M2=np.exp(logM2)
  M0=np.exp(a0+b0*T+c0*logM2)
  M3=np.exp(a3+b3*T+c3*logM2)

  rhos=0.069*6/(D*np.pi)

  Deqi=D*(rhoi/917)**(1/3.)
  Deqs=D*(rhos/917)**(1/3.)
  Deqg=D*(rhog/917)**(1/3.)
  Deqi_mid=Deqi[0:-1]+np.diff(Deqi)/2.
  Deqs_mid=Deqs[0:-1]+np.diff(Deqs)/2.
  Deqg_mid=Deqg[0:-1]+np.diff(Deqg)/2.
  dDeqi=np.diff(Deqi)
  dDeqs=np.diff(Deqs)
  dDeqg=np.diff(Deqg)

  def ns_dist(x,M2,M3,Kap0,Kap1,Lam0,Lam1,mu_s,rho):
    return 1e-6*M2**4/M3**3*(Kap0*np.exp(-M2*Lam0*x/M3)+\
           Kap1*(M2/M3)**mu_s*x**mu_s*np.exp(-M2*Lam1*x/M3))*rho

  def ncirg_dist(x,n0,lam,mu,rho):
    return n0*x**mu*np.exp(-lam*x)*1e3

  distc_array=ncirg_dist(D_mid,N0_c,lamc,mu_c,rho)*1e6
  disti_array=ncirg_dist(D_mid,N0_i,lamc,mu_i,rho)*1e6
  distr_array=ncirg_dist(D_mid,N0_r,lamr,mu_r,rho)*1e6
  distg_array=ncirg_dist(D_mid,N0_g,lamg,mu_g,rho)*1e6
  dists_array=ns_dist(D_mid,M2,M3,Kap0,Kap1,Lam0,\
                      Lam1,mu_s,rho)*1e6

  return D_mid,Deqi_mid,D_mid,Deqs_mid,Deqg_mid,\
         dD,dDeqi,dD,dDeqs,dDeqg,\
         distc_array,disti_array,distr_array,\
         dists_array,distg_array 

####################################################
#Compute reflectivity factors and attenuation coefficients
####################################################
def zeff(N,Deq_mid,dDeq,T,m0,ice=False):
  rhoi=917.
  freq=94.4
  k2=.75
  wl = 2.99792458/(freq*10)
  sizep = (np.pi*Deq_mid)/wl
  qext=np.zeros(len(sizep))
  qbsca=np.zeros(len(sizep))
  qsca=np.zeros(len(sizep))

  #Use external mie scatering code bhmie.py
  for i in range(len(sizep)):
    [_,_,qext[i],qsca[i],qbsca[i],_]=\
    bhmie.bhmie(sizep[i],m0,2) 
  cr=10./np.log(10)

  #Compute unattenuated effective reflectivity factor
  eta_sum=np.sum(qbsca*np.pi*N*(Deq_mid/2)**2.*dDeq)
  eta_mie=eta_sum*.25
  z_eff=(wl**4/np.pi**5)*(1./k2)*eta_mie*1e18

  #Compute attenuation coefficient
  k_sum=np.sum(qext*np.pi*N*Deq_mid**2.*dDeq)
  kr=k_sum*.25*np.pi*1000.*cr

  #Compute reflectivity factor, Rayleigh only
  z_ray=np.sum(Deq_mid**6.*N*dDeq)*1e18

  return kr,z_eff,z_ray

####################################################
#Compute index of refraction for liquid and ice#####
####################################################

#Refractive index of liquid water from QUICKBEAM
def compute_m_wat(freq,tk):
  freq=94.
  tc = tk - 273.15
  ld = 100.*2.99792458E8/(freq*1E9)
  es = 78.54*(1-(4.579E-3*(tc-25.)+1.19E-5*(tc-25.)**2\
       -2.8E-8*(tc-25.)**3))
  ei = 5.27137+0.021647*tc-0.00131198*tc**2
  a = -(16.8129/(tc+273.))+0.0609265
  ls = 0.00033836*np.exp(2513.98/(tc+273.))
  sg = 12.5664E8

  tm1 = (ls/ld)**(1-a)
  pi = np.arccos(-1.)
  cos1 = np.cos(0.5*a*np.pi)
  sin1 = np.sin(0.5*a*np.pi)

  e_r = ei + (((es-ei)*(1.+tm1*sin1))/\
             (1.+2*tm1*sin1+tm1**2))
  e_i = (((es-ei)*tm1*cos1)/(1.+2*tm1*sin1+tm1**2))\
        +((sg*ld)/1.885E11)

  e_comp = complex(e_r,e_i)
  sq = cmath.sqrt(e_comp)

  n_r = np.real(sq)
  n_i = np.imag(sq)
  return n_r,n_i

#Refractive index of ice from QUICKBEAM
def compute_m_ice(freq,tk):
  freq=94.
  alam=3E5/freq
  with open('m_ice_tables','rb') as fid:
    Lib=pickle.load(fid)
  tabret=Lib['tabret']
  tabimt=Lib['tabimt']
  wlt=Lib['wlt']

  temref=[272.16,268.16,253.16,213.16]

  if tk>272.16:
    tk=272.16
  if tk<213.16:
    tk=213.16
  worked=False
  for i in range(3):
    if tk>temref[i+1]:
      lt2=i
      lt1=i-1
      for j in range(61):
        if alam<wlt[j+1]:
          x1=np.log(wlt[j])
          x2=np.log(wlt[j+1])
          y1=tabret[j+lt1*61]
          y2=tabret[j+lt1*61]
          x=np.log(alam)
          ylo=((x-x1)*(y2-y1)/(x2-x1))+y1
          y1=tabret[j+lt2*61]
          y2=tabret[j+lt2*61+1]
          yhi=((x-x1)*(y2-y1)/(x2-x1))+y1
          t1=temref[lt1]
          t2=temref[lt2]
          y=((tk-t1)*(yhi-ylo)/(t2-t1))+ylo
          n_r=y
          y1=np.log(abs(tabimt[j+lt1*61]))
          y2=np.log(abs(tabimt[j+lt1*61+1]))
          ylo=((x-x1)*(y2-y1)/(x2-x1))+y1
          y1=np.log(abs(tabimt[j+lt2*61]))
          y2=np.log(abs(tabimt[j+lt2*61+1]))
          yhi=((x-x1)*(y2-y1)/(x2-x1))+y1
          y=((tk-t1)*(yhi-ylo)/(t2-t1))+ylo
          n_i=np.exp(y)
          worked=True
          break
  if not(worked):
    n_r=np.nan
    n_i=np.nan
  return n_r,n_i

