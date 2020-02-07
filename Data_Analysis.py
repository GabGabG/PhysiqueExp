import matplotlib.pyplot as plt
import numpy as np
from file_readers import get_reader
from scipy.optimize import curve_fit


class FitGaussian:
    def __init__(self, data_x, data_y, a, b, energy_th):
        self.x_values = data_x
        self.y_values = data_y

        self.picx = self.x_values[a:b]
        self.picy = self.y_values[a:b]
        # parametres_pics: [a, b, amplitude fit gaussien, moyenne fit gaussien, écart type fit gaussien,
        # FWHM (résolution absolue), résolution relative, Energie théorique]
        self.parametres_pics = np.array([a, b, 0, 0, 0, 0, 0, energy_th], dtype=float)
        self.bckParam = None
        self.gaussianfit()

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
        bckg = (popt[0] * self.picx + popt[1])
        pic_noiseModif = self.picy + bckg if action.strip() == "+" \
            else self.picy - bckg
        self.bckParam = popt
        return pic_noiseModif

    @staticmethod
    def gaus(x, a, mu, sigma):
        return a * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))

    def gaussianfit(self):
        mean = np.mean(self.picx)
        sigma = np.std(self.picx)
        picy_wonoise = self.bkghandling("-")
        popt, pcov = curve_fit(self.gaus, self.picx, picy_wonoise, p0=[1, mean, sigma])
        # TODO: à voir
        picy_wonoise = self.bkghandling("+")
        self.parametres_pics[5] = 2 * np.sqrt(2 * np.log(2)) * abs(popt[2])
        self.parametres_pics[2] = popt[0]
        self.parametres_pics[3] = popt[1]
        self.parametres_pics[4] = abs(popt[2])
        self.parametres_pics[6] = self.parametres_pics[5] / self.parametres_pics[3] * 100
        print(self.parametres_pics[6])


class Photopics:

    def __init__(self, filename: str, pics_info: tuple):
        # pics_info se veut comme une liste de tuples avec chaque tuple contenant l'info suivante:
        # (a, b, énergie théorique)
        self.filename = filename
        self.pics_info = pics_info
        reader = get_reader(filename)
        reader.find_histogram_data()
        self.x_values = reader.get_x_indices()
        self.y_values = reader.histogram_data
        self.photopics_params, self.bckgs = self.getGaussianFitInfo()

    def getGaussianFitInfo(self):
        photopics_params_temp = []
        bckgs = []
        for params in self.pics_info:
            a, b, energie_theorique = params
            gaussianInfoTemp = FitGaussian(self.x_values, self.y_values, a, b, energie_theorique)
            photopics_params_temp.append(gaussianInfoTemp.parametres_pics)
            bckgs.append(gaussianInfoTemp.bckParam)
        return np.vstack(photopics_params_temp), bckgs


class Detector:

    def __init__(self, filelist_and_params: list, detecteur: str):
        self.detecteur = detecteur
        self.params_data_array = []
        self.x_values = []
        self.y_values = []
        self.bckgs = []
        self.sources = []
        for file in filelist_and_params:
            photopics = Photopics(file[0], file[-1])
            self.params_data_array.append(photopics.photopics_params)
            self.bckgs.append(photopics.bckgs)
            self.x_values.append(photopics.x_values)
            self.y_values.append(photopics.y_values)
            self.sources.append(file[1])

        self.params_data_array = np.vstack(self.params_data_array)
        self.bckgs = np.vstack(self.bckgs)
        self.calibation_params = None

    def showSpectraWithGaussianFit(self, spectrum_colors: list, fit_colors: list):
        # TODO: Couleur
        a_s = self.params_data_array[:, 0]
        b_s = self.params_data_array[:, 1]
        A_s = self.params_data_array[:, 2]
        mu_s = self.params_data_array[:, 3]
        sigma_s = self.params_data_array[:, 4]

        for i in range(len(self.x_values)):
            source = self.sources[i]
            atomic_number, symbol = source.split(" ")
            color = spectrum_colors[i]
            # TODO: Exposant / numéro atomique
            plt.bar(self.x_values[i], self.y_values[i], width=1, alpha=0.5,
                    label=r"Spectre du ${}^{}{}$".format("", atomic_number, symbol), color=color)

        # Comme tous les sub arrays ont la même longueur, on s'en fout
        for i in range(len(a_s)):
            color = fit_colors[i]
            a = a_s[i]
            b = b_s[i]
            A = A_s[i]
            mu = mu_s[i]
            sigma = sigma_s[i]
            bckg = self.bckgs[i]
            range_x_gauss = np.linspace(a, b, 10000)
            plt.plot(range_x_gauss, FitGaussian.gaus(range_x_gauss, A, mu, sigma) + bckg[0] * range_x_gauss + bckg[1],
                     label=r"Lissage gaussien: $\mu = {}$, $\sigma = {}$".format(mu, sigma), color=color)
        plt.legend()
        plt.xlabel("Canal [-]")
        plt.ylabel("Nombre de compte [-]")
        plt.title(f"Spectre énergétique du détecteur {self.detecteur}.")
        plt.show()

    def calibrationShow(self):
        energies = self.params_data_array[:, -1]
        centroids = self.params_data_array[:, 3]
        plt.scatter(centroids, energies)
        popt, pcov = curve_fit(lambda x, a, b: a * x + b, centroids, energies, p0=[1, 0])
        self.calibation_params = popt
        x_calib = np.linspace(min(centroids), max(centroids), 1000)
        plt.plot(x_calib, popt[0] * x_calib + popt[-1], label="Régression linéaire.", linestyle="dashed")
        plt.legend()
        plt.ylabel("Énergie du photopic [keV]")
        plt.xlabel("Position du centroïd [-]")
        plt.title(f"Régression linéaire pour la calibration en énergie du détecteur {self.detecteur}.")
        plt.show()

    def conversion_Channel_EnergyShow(self):
        # TODO: Couleur
        x_values_energy = self.x_values * self.calibation_params[0] + self.calibation_params[-1]
        plt.bar(x_values_energy, self.y_values, width=1)
        plt.ylabel("Nombre de compte [-]")
        plt.xlabel("Énergie [keV]")
        plt.title(f"Spectre en énergie calibré pour le détecteur {self.detecteur}.")
        plt.show()

    def resolutionShow(self):
        FWHMs = self.params_data_array[:, 5]
        energies = self.params_data_array[:, -1]
        relative_resolution = self.params_data_array[:, 6]
        plt.scatter(energies, FWHMs)
        plt.show()
        plt.scatter(energies, relative_resolution)
        plt.show()


if __name__ == '__main__':
    detecteur = "NaI(Tl)"
    path1 = r"C:\Users\goubi\PycharmProjects\PhysiqueExp\PhysiqueExp\Intro_Gamma\Etalonnage_22Na.Spe"
    path2 = r"C:\Users\goubi\PycharmProjects\PhysiqueExp\PhysiqueExp\Intro_Gamma\Etalonnage_57Co.Spe"
    path3 = r"C:\Users\goubi\PycharmProjects\PhysiqueExp\PhysiqueExp\Intro_Gamma\Etalonnage_60Co.Spe"
    path4 = r"C:\Users\goubi\PycharmProjects\PhysiqueExp\PhysiqueExp\Intro_Gamma\Etalonnage_137Cs.Spe"
    # reader = get_reader(path4)
    # reader.find_histogram_data()
    # y_val = reader.histogram_data
    # x_val = reader.get_x_indices()
    # plt.bar(x_val, y_val, width=1)
    # plt.show()
    filesNaI = [[path1, "22 Na", [(3016, 3376, 1275), (1184, 1484, 511)]], [path2, "57 Co", [(295, 400, 122)]],
                [path3, "60 Co", [(3141, 3542, 1333), (2800, 3118, 1173)]],
                [path4, "137 Cs", [(1546, 1880, 662), (59, 116, 30)]]]
    # spectrum_color = ["#1DE4BD", "#1AC9E6", "#19AADE", "#176BA0"]
    # fit_colors = ["#1DE4BD", "#1DE4BD", "#1AC9E6", "#19AADE", "#19AADE", "#176BA0", "#176BA0"]
    spectrum_color = ["#EB548C", "#7D3AC1", "#EA7369", "#AF4BCE"]
    fit_colors = ["#EB548C", "#EB548C", "#7D3AC1", "#EA7369", "#EA7369", "#AF4BCE", "#AF4BCE"]
    d = Detector(filesNaI, detecteur)
    d.showSpectraWithGaussianFit(spectrum_color, fit_colors)
    d.calibrationShow()
    d.resolutionShow()
