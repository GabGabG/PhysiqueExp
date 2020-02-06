import matplotlib.pyplot as plt
import numpy as np
from file_readers import get_reader
from scipy.optimize import curve_fit
import pandas as pd


class FitGaussian:
    def __init__(self, data_x, data_y, a, b):

        self.x_values = data_x
        self.y_values = data_y

        self.picx = self.x_values[a:b]
        self.picy = self.y_values[a:b]
        # parametres_pics: [Énergie, a, b, amplitude fit gaussien, moyenne fit gaussien, écart type fit gaussien,
        # FWHM (résolution absolue), résolution relative]
        self.parametres_pics = np.array([0, a, b, 0, 0, 0, 0, 0])
        self.FWHM = None
        self.centroid = None
        self.popt, self.pcov = self.gaussianfit()

    def bkghandling(self, action):
        if action.strip() not in ["+", "-"]:
            raise Exception("Invalid action, must be '+' or '-'")
        xa = self.picx[:5]
        xb = self.picx[-5:]
        x_bkg = np.append(xa, xb)
        ya = self.picy[:5]
        yb = self.picy[-5:]
        y_bkg = np.append(ya, yb)
        popt, pcov = curve_fit(lambda x, a, b: a * x + b, x_bkg, y_bkg, p0=[0, 0])
        pic_noiseModif = self.picy + (popt[0] * self.picx + popt[1]) if action.strip() == "+" \
            else self.picy - (popt[0] * self.picx + popt[1])
        return pic_noiseModif

    @staticmethod
    def gaus(x, a, mu, sigma):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def gaussianfit(self):
        mean = np.mean(self.picx)
        sigma = np.std(self.picx)
        picy_wonoise = self.bkghandling("-")
        popt, pcov = curve_fit(self.gaus, self.picx, picy_wonoise, p0=[1, mean, sigma])
        picy_wonoise = self.bkghandling("+")
        self.popt = popt
        self.FWHM = 2 * np.sqrt(2 * np.log(2)) * popt[2]
        self.centroid = popt[1]
        self.parametres_pics[3] = popt[0]
        self.parametres_pics[4] = popt[1]
        self.parametres_pics[5] = popt[2]
        self.parametres_pics[6] = self.resolutionabs()
        self.parametres_pics[7] = self.resolutionrelative()
        return popt, pcov

    def resolutionabs(self):
        if self.FWHM is None:
            raise Exception("Requires gaussianfit first")
        return self.FWHM

    def resolutionrelative(self):
        if self.FWHM is None and self.centroid is None:
            raise Exception("Requires gaussianfit first")
        return self.FWHM / self.centroid


class Master:
    def __init__(self, fileseries):
        pass

    def fitcalibration_energie(self):
        pass

    def spectres(self):
        pass


class Photopics:

    def __init__(self, filename: str, pics_info: list):
        # pics_info se veut comme une liste de tuples avec chaque tuple contenant l'info suivante:
        # (a, b, énergie théorique)
        self.filename = filename
        self.pics_info = pics_info
        reader = get_reader(filename)
        reader.find_histogram_data()
        self.x_values = reader.get_x_indices()
        self.y_values = reader.histogram_data
        self.gaussianParams = self.getGaussianFitInfo()

    def getGaussianFitInfo(self):
        gaussianInfo = []
        for params in self.pics_info:
            a, b, energie_theorique = params
            gaussianInfoTemp = FitGaussian(self.x_values, self.y_values, a, b)
            gaussianInfo.append(gaussianInfoTemp.parametres_pics)
        return gaussianInfo

    def linearFit_calibration(self, energies, channels):
        f = lambda x, a, b: a * x + b
        popt, pcov = curve_fit(f, channels, energies, p0=[1, 0])
        return popt, pcov

    def showSpectraWithGaussianFit(self, detecteur: str):
        plt.bar(self.x_values, self.y_values, width=1)
        for photopic in self.gaussianParams:
            a = self.pics_info[0]
            b = self.pics_info[1]
            sigma = photopic[5]
            mu = photopic[4]
            A = photopic[3]
            x = np.linspace(a, b, 1000)
            plt.plot(x, FitGaussian.gaus(x, A, mu, sigma),
                     label=r"Fit gaussien avec $\sigma = {}$, $\mu = {}$".format(sigma, mu))
        plt.legend()
        plt.title(f"Spectre d'énergie pour le détecteur {detecteur}")
        plt.xlabel("Canal [-]")
        plt.ylabel("Nombre de compte [-]")
        plt.show()

    def calibrationShow(self):
        energies = [e[-1] for e in self.pics_info]
        centroids = [c[4] for c in self.gaussianParams]
        plt.scatter(centroids, energies, label="Position des centroïdes approximée par le lissage gaussien")
        popt, pcov = self.linearFit_calibration(energies, centroids)
        slope = popt[0]
        x0 = popt[-1]
        x = np.linspace(min(centroids), max(centroids), 1000)
        plt.plot(x, slope * np.array(centroids) + x0, linestyle="dashed")
        plt.show()

