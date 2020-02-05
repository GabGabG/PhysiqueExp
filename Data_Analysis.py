import matplotlib.pyplot as plt
import numpy as np
from file_readers import get_reader
from os import listdir, path
from scipy.optimize import curve_fit
import pandas as pd


class FitGaussian:
    def __init__(self, filename, a, b):
        self.data_handler = get_reader("Data Lab Spectro/NaI(TI)/" + filename)

        self.x_values = data_handler.get_x_indices()
        self.y_values = data_handler.histogram_data

        self.picx = self.x_values[a:b]
        self.picy = self.y_values[a:b]

        self.parametres_pics = np.array([0, a, b, 0, 0, 0, 0, 0])

        self.popt = None
        self.FWHM = None
        self.centroid = None

    def bkghandling(self, action):
        if action.strip() not in ["+", "-"]:
            raise Exception("Invalid action, must be + or -")
        xa = self.picx[:5]
        xb = self.picx[-5:]
        x_bkg = np.append(xa, xb)
        ya = self.picy[:5]
        yb = self.picy[-5:]
        y_bkg = np.append(ya, yb)
        popt, pcov = sp.optimize.curve_fit(lambda x, a, b : a*x+b, x_bkg, y_bkg, p0=[0, 0])
        pic_wonoise = self.picy + (popt[0] * self.picx + popt[1]) if action.strip() == "+" \
            else self.picy - (popt[0] * self.picx + popt[1])
        return pic_wonoise

    def gaus(self,x,a,x0,sigma):
        return a*np.exp(-(x-x0)**2/(2*sigma**2))

    def gaussianfit(self):
        mean = np.mean(self.picx)
        sigma = np.std(self.picx)
        picy_wonoise = self.bkghandling(self.picy, "-")
        popt, pcov = curve_fit(self.gaus, self.picx, picy_wonoise, p0=[1, mean, sigma])
        picy_wonoise = self.bkghandling(self.picy, "+")
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
        return self.FWHM/self.centroid




class Master:
    def __init__(self, detecteur, fileserie):

        self.conversionenergie = []


        self.param_detecteur = []
        for i in detecteur:
            for j in param_detecteur:
            self.param_detecteur += [dataframe_param]
        #todo hihi



    @staticmethod
    def __data_dataframe_generator(detecteur, filename):
        self.data_handler = get_reader("Data Lab Spectro/" + detecteur  "/" + filename)
        self.data_handler.find_histogram_data()
        data = self.data_handler.histogram_data
        df = pd.DataFrame(data, columns=["x", "y"])
        #todo deal avec le fait que doit label de detecteur et le filename sur le dataframe pour gerer les spectres avec plusieurs pics
        return

    def fitcalibration_energie(self):
        for i in self.param_detecteur:
            popt, pcov = sp.optimize.curve_fit(lambda x, a, b: a * x + b, self.dataframe_param.centroid, self.dataframe_param.energieTheo, p0=[1, 0])
            self.conversionenergie.append(popt)
            plt.scatter(self.dataframe_param.centroid, self.dataframe_param.energieTheo)
            x = np.arange(min(self.dataframe_param.centroid), max(self.dataframe_param.centroid))
            plt.plot(x, popt[0]*x+popt[1])
            #todo legend
        plt.xlabel("Canaux [-]")
        plt.ylabel("Énergie [keV]")
        plt.title("Énergie des pics en fonction des canaux des différents détecteurs")
        plt.show()


    def spectres(self):
        pass




