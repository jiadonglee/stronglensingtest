# credit Nan Li 2020.
import copy
import numpy as np
import jax.numpy as jnp
from astropy import constants as const
import astropy.units as u
from astropy.cosmology import FlatwCDM
import matplotlib.pyplot as plt


cosmo = FlatwCDM(H0=71, Om0=0.264, Ob0=0.044792699861138666, w0=-1.000000)

vc = const.c.to(u.km/u.second)  # km/s
apr = u.rad.to(u.arcsecond)  # 1.0/np.pi*180.*3600        # 1/1^{''}


def make_r_coor(bs, nc):
    ds = bs/nc
    xx01 = jnp.linspace(-bs/2.0, bs/2.0-ds, nc)+0.5*ds
    xx02 = jnp.linspace(-bs/2.0, bs/2.0-ds, nc)+0.5*ds
    xg1, xg2 = jnp.meshgrid(xx01, xx02)
    return xg1, xg2


class StrongLensingSim(object):
    def __init__(self, xg1, xg2, lensCat, srcsCat, cosmology=cosmo):
        # meshgrids
        self.xg1 = xg1
        self.xg2 = xg2
        # cosmological model
        self.cosmology = cosmology
        self.lensCat = copy.deepcopy(lensCat)
        self.srcsCat = copy.deepcopy(srcsCat)
        # lensing simulation
        self.create_lens_model()
        self.ray_shooting()
        self.generate_lensed_sersic()
        # self.add_noise() # where you do modeling, you do not need to include this step

    def create_lens_model(self):
        # create mass model of lens and calculate deflection angles
        Das = jnp.array(self.cosmology.angular_diameter_distance(
            self.srcsCat['ZSRC']).value)
        Dals = jnp.array(self.cosmology.angular_diameter_distance_z1z2(
            self.lensCat['ZLENS'], self.srcsCat['ZSRC']).value)

        self.lensCat['REIN'] = 4.0*jnp.pi * \
            (self.lensCat['VELDISP']/vc.value)**2.0*Dals/Das*apr
        sx = self.xg1 - self.lensCat['XLENS']
        sy = self.xg2 - self.lensCat['YLENS']
        cs = jnp.cos(self.lensCat['PHIE'])
        sn = jnp.sin(self.lensCat['PHIE'])
        sx_r = sx * cs + sy * sn
        sy_r = -sx * sn + sy * cs
        ql = 1.0-self.lensCat['ELLIP']
        fq = 1.0/(jnp.sqrt((1.+ql**2)/(2. * ql)))*jnp.sqrt((1 + ql**2)/2)
        reScale = self.lensCat['REIN']*fq
        rcScale = self.lensCat['RCORE']*jnp.sqrt((1 + ql**2) / (2*ql**2))
        psi = jnp.sqrt(ql**2.0*(rcScale**2.0+sx_r**2.0)+sy_r**2.0)

        # if ql==1.0:
        #     dx_r = reScale*sx_r/(psi+rcScale)
        #     dy_r = reScale*sy_r/(psi+rcScale)
        # else:
        dx_r = (reScale/jnp.sqrt((1.0-ql**2.0))) * \
            jnp.arctan(jnp.sqrt((1.0-ql**2.0))*sx_r/(psi + rcScale))
        dy_r = (reScale/jnp.sqrt((1.0-ql**2.0))) * \
            jnp.arctanh(jnp.sqrt((1.0-ql**2.0))*sy_r/(psi + rcScale*ql**2.0))

        # primary lens
        dx = dx_r * cs - dy_r * sn
        dy = dx_r * sn + dy_r * cs
        # external shear
        tr2 = self.lensCat['PHIG']
        cs2 = jnp.cos(2.0 * tr2)
        sn2 = jnp.sin(2.0 * tr2)
        dx2 = self.lensCat['GAMMA'] * (cs2 * sx + sn2 * sy)
        dy2 = self.lensCat['GAMMA'] * (sn2 * sx - cs2 * sy)
        # external kappa
        dx3 = self.lensCat['EXTKAPPA'] * sx
        dy3 = self.lensCat['EXTKAPPA'] * sy
        # total
        self.alpha1 = dx + dx2 + dx3
        self.alpha2 = dy + dy2 + dy3

    def ray_shooting(self):
        # shooting light rays from lens plane to source plane
        self.yg1 = self.xg1 - self.alpha1
        self.yg2 = self.xg2 - self.alpha2

    def generate_lensed_sersic(self):
        # generate lensed images of a source modeled by Sersic profile
        bn = 2.0*self.srcsCat['SINDEX']-1/3.0+0.009876/self.srcsCat['SINDEX']
        xi1new = (self.yg1-self.srcsCat['XSRC'])*jnp.cos(self.srcsCat['PHIS'])+(
            self.yg2-self.srcsCat['YSRC'])*jnp.sin(self.srcsCat['PHIS'])
        xi2new = (self.yg2-self.srcsCat['YSRC'])*jnp.cos(self.srcsCat['PHIS'])-(
            self.yg1-self.srcsCat['XSRC'])*jnp.sin(self.srcsCat['PHIS'])
        R_scale = jnp.sqrt(
            (xi1new/self.srcsCat['ASRC'])**2+(xi2new/self.srcsCat['BSRC'])**2)
        
        R_scale_th = 0.01
        # R_scale[R_scale<R_scale_th] = R_scale_th

        # idx = R_scale<R_scale_th
        # R_scale.at[idx].set(R_scale_th)

        self.img = jnp.exp(-bn*((R_scale)**(1.0/self.srcsCat['SINDEX'])-1.0))
        self.lensed_images = self.img / \
            jnp.exp(-bn*((R_scale_th)**(1.0/self.srcsCat['SINDEX'])-1.0))
        self.xi1new = xi1new


    def add_noise(self):
        # add noise to the clean simulated images
        nx, ny = jnp.shape(self.lensed_images)
        mu, sigma = 0, jnp.max(self.lensed_images)*0.1
        noise_map = jnp.random.normal(mu, sigma, (nx, ny))
        self.lensed_images += noise_map
        self.lensed_images_lin = self.lensed_images.reshape(-1)
        self.noise_map = noise_map.reshape(-1)

    def visualize_lensed_images(self):
        # visualize the output
        plt.figure(figsize=(8, 8))
        plt.imshow(self.lensed_images)

def make_r_coor_numpy(bs, nc):
    ds = bs/nc
    xx01 = np.linspace(-bs/2.0, bs/2.0-ds, nc)+0.5*ds
    xx02 = np.linspace(-bs/2.0, bs/2.0-ds, nc)+0.5*ds
    xg1, xg2 = np.meshgrid(xx01, xx02)
    return xg1, xg2


class StrongLensingSim_numpy(object):
    def __init__(self, xg1, xg2, lensCat, srcsCat, cosmology=cosmo):
        # meshgrids
        self.xg1 = xg1
        self.xg2 = xg2
        # cosmological model
        self.cosmology = cosmology
        self.lensCat = copy.deepcopy(lensCat)
        self.srcsCat = copy.deepcopy(srcsCat)
        # lensing simulation
        self.create_lens_model()
        self.ray_shooting()
        self.generate_lensed_sersic()
        # self.add_noise() # where you do modeling, you do not need to include this step

    def create_lens_model(self):
        # create mass model of lens and calculate deflection angles
        Das = np.array(self.cosmology.angular_diameter_distance(
            self.srcsCat['ZSRC']).value)
        Dals = np.array(self.cosmology.angular_diameter_distance_z1z2(
            self.lensCat['ZLENS'], self.srcsCat['ZSRC']).value)

        self.lensCat['REIN'] = 4.0*np.pi * \
            (self.lensCat['VELDISP']/vc.value)**2.0*Dals/Das*apr
        sx = self.xg1 - self.lensCat['XLENS']
        sy = self.xg2 - self.lensCat['YLENS']
        cs = np.cos(self.lensCat['PHIE'])
        sn = np.sin(self.lensCat['PHIE'])
        sx_r = sx * cs + sy * sn
        sy_r = -sx * sn + sy * cs
        ql = 1.0-self.lensCat['ELLIP']
        fq = 1.0/(np.sqrt((1.+ql**2)/(2. * ql)))*np.sqrt((1 + ql**2)/2)
        reScale = self.lensCat['REIN']*fq
        rcScale = self.lensCat['RCORE']*np.sqrt((1 + ql**2) / (2*ql**2))
        psi = np.sqrt(ql**2.0*(rcScale**2.0+sx_r**2.0)+sy_r**2.0)

        dx_r = (reScale/np.sqrt((1.0-ql**2.0))) * \
            np.arctan(np.sqrt((1.0-ql**2.0))*sx_r/(psi + rcScale))
        dy_r = (reScale/np.sqrt((1.0-ql**2.0))) * \
            np.arctanh(np.sqrt((1.0-ql**2.0))*sy_r/(psi + rcScale*ql**2.0))

        # primary lens
        dx = dx_r * cs - dy_r * sn
        dy = dx_r * sn + dy_r * cs
        # external shear
        tr2 = self.lensCat['PHIG']
        cs2 = np.cos(2.0 * tr2)
        sn2 = np.sin(2.0 * tr2)
        dx2 = self.lensCat['GAMMA'] * (cs2 * sx + sn2 * sy)
        dy2 = self.lensCat['GAMMA'] * (sn2 * sx - cs2 * sy)
        # external kappa
        dx3 = self.lensCat['EXTKAPPA'] * sx
        dy3 = self.lensCat['EXTKAPPA'] * sy
        # total
        self.alpha1 = dx + dx2 + dx3
        self.alpha2 = dy + dy2 + dy3

    def ray_shooting(self):
        # shooting light rays from lens plane to source plane
        self.yg1 = self.xg1 - self.alpha1
        self.yg2 = self.xg2 - self.alpha2

    def generate_lensed_sersic(self):
        # generate lensed images of a source modeled by Sersic profile
        bn = 2.0*self.srcsCat['SINDEX']-1/3.0+0.009876/self.srcsCat['SINDEX']
        xi1new = (self.yg1-self.srcsCat['XSRC'])*np.cos(self.srcsCat['PHIS'])+(
            self.yg2-self.srcsCat['YSRC'])*np.sin(self.srcsCat['PHIS'])
        xi2new = (self.yg2-self.srcsCat['YSRC'])*np.cos(self.srcsCat['PHIS'])-(
            self.yg1-self.srcsCat['XSRC'])*np.sin(self.srcsCat['PHIS'])
        R_scale = np.sqrt(
            (xi1new/self.srcsCat['ASRC'])**2+(xi2new/self.srcsCat['BSRC'])**2)
        
        R_scale_th = 0.01

        self.img = np.exp(-bn*((R_scale)**(1.0/self.srcsCat['SINDEX'])-1.0))
        self.lensed_images = self.img / \
            np.exp(-bn*((R_scale_th)**(1.0/self.srcsCat['SINDEX'])-1.0))
        self.xi1new = xi1new
    
    def visualize_lensed_images(self):
        # visualize the output
        plt.figure(figsize=(8, 8))
        plt.imshow(self.lensed_images)