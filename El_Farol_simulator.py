import csv
import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit


class Strategy:
    """
    A class that represents a strategy for predicting attendance at El Farol
    """

    def __init__(self, M):
        """
        Initializes the strategy with M random coefficients between -1 and 1
        """
        self.coeffs = np.random.uniform(-1, 1, M)

    # def get_strategies_score(self, actual_attendance):
    #     """
    #     Gets the score after actual attendance
    #     """
    #     return np.sum(np.abs(self.scores - actual_attendance[-self.M:][::-1]))

    def predict(self, past_attendance):
        """
        Predicts the attendance for the next week based on the given past attendance
        """
        return np.dot(self.coeffs, past_attendance)


class Person:
    """
    A class that represents a person attending El Farol
    """

    def __init__(self, K, M, T):
        """
        Initializes the person with K strategies, each with M coefficients,
        and a threshold T above which staying at the club is unpleasant
        """
        self.K = K
        self.M = M
        self.T = T
        self.strategies = [Strategy(M) for _ in range(K)]
        self.scores = np.zeros(K)

    # def choose_strategy(self, past_attendance):
    #     """
    #     Chooses the best strategy based on past attendance
    #     """
    #     predicted_attendances = [strategy.predict(past_attendance) for strategy in self.strategies]
    #     differences = [abs(predicted_attendance - past_attendance[-1]) for predicted_attendance in
    #                    predicted_attendances]
    #     best_strategy_index = np.argmin(differences)
    #     return self.strategies[best_strategy_index]

    def choose_strategy(self, past_attendance):
        """
        Chooses the best strategy based on past attendance
        """
        predicted_attendances = [strategy.predict(past_attendance) for strategy in self.strategies]
        differences = [np.sqrt(np.mean((predicted_attendance - past_attendance) ** 2))
                       for predicted_attendance in predicted_attendances]
        best_strategy_index = np.argmin(differences)
        return self.strategies[best_strategy_index]

    def update_scores(self, actual_attendance):
        """
        Updates the scores of all strategies based on the actual attendance
        """
        past_attendance = actual_attendance[-self.M:]
        # print(self.M, past_attendance)
        predicted_attendance = np.array([strategy.predict(past_attendance) for strategy in self.strategies])
        errors = (predicted_attendance - actual_attendance[-1]) ** 2
        for i, error in enumerate(errors):
            self.scores[i] = (self.scores[i] * (self.M - 1) + error) / self.M

    def should_attend(self, past_attendance):
        """
        Determines whether the person should attend based on past attendance
        and the best strategy
        """
        best_strategy = self.choose_strategy(past_attendance)
        predicted_attendance = abs(best_strategy.predict(past_attendance))
        # if self.K < self.M:
        #     return predicted_attendance > self.T * np.mean(past_attendance[-2:])
        return predicted_attendance > self.T * np.mean(past_attendance[-self.K:])


class ElFarol:
    """
    A class that represents El Farol bar
    """

    def __init__(self, N, K, M, T=0.6):
        """
        Initializes El Farol with N people, each with K strategies, each with M coefficients,
        and a threshold T above which staying at the club is unpleasant
        """
        self.N = N
        self.K = K
        self.M = M
        self.T = T
        self.people = [Person(K, M, T) for _ in range(N)]
        self.attendance = np.zeros(M)
        self.mean_attendance = None
        self.std_attendance = None
        self.time_to_converge = None

    def run(self, num_weeks):
        """
        Runs El Farol for the given number of weeks
        """
        # self.predicted_attendance_for_person_to_follow = np.zeros(num_weeks)
        for week in range(num_weeks):
            # determine attendance for this week
            if week == 0:
                # first week, attendance is random
                actual_attendance = np.random.randint(self.N)
            else:
                # subsequent weeks, each person decides whether to attend or not
                actual_attendance = sum(
                    [person.should_attend(self.attendance[-self.M:]) for person in self.people])

            # add predicted attendance to person to follow
            # if self.people[self.person_to_follow].should_attend(self.attendance[-self.M:]):
            #     self.predicted_attendance_for_person_to_follow[week] = 1
            # else:
            #     self.predicted_attendance_for_person_to_follow[week] = 0

            # add actual attendance to attendance record
            self.attendance = np.append(self.attendance, actual_attendance)

            # update scores and choose new strategies
            for person in self.people:
                person.update_scores(self.attendance)
                person.strategies[np.argmin(person.scores)] = Strategy(self.M)
        self.mean_attendance = np.mean(self.attendance)
        self.std_attendance = np.std(self.attendance)
        self.time_to_converge = np.argmin(
                    np.abs(self.attendance - np.mean(self.attendance)))

    def plot_attendance(self):
        """
        Plots the attendance over time
        """
        plt.figure()
        plt.plot(range(len(self.attendance)), self.attendance, label="Attendance Over Weeks")
        plt.axhline(y=(int(self.T * self.N)), color='r', label=f"Threshold {int(self.T * self.N)}")
        plt.legend()
        plt.xlabel(r"$Weeks\ [A.U.]$")
        plt.ylabel(r"$Attendance\ [A.U.]$")
        plt.title(f"Bar Attendance VS Weeks, K={self.K}, M={self.M}")
        plt.savefig(os.path.join(basic_plots_path + attendance_plots_dir, f"K {self.K}__M {self.M}__{plot_format}"))
        # plt.show()
        plt.close()


class Simulation:
    """
    A class that runs multiple simulations of the ElFarol class
    """

    def __init__(self, N, num_weeks, K_range, M_range, plot=True):
        """
        Initializes the simulation with N people and the given number of weeks
        """
        self.N = N
        self.num_weeks = num_weeks
        self.K_range = K_range
        self.M_range = M_range
        self.plot = plot
        self.attendance = np.empty((K_range, M_range), dtype=object)
        self.mean_attendance = np.zeros((K_range, M_range))
        self.std_attendance = np.zeros((K_range, M_range))
        self.mean_convergence_time = np.zeros((K_range, M_range))
        self.run_simulations()

    def run_simulations(self):
        """
        Runs simulations of the ElFarol class with different values of K and M
        """
        take_care_of_dirs(attendance_plots_dir, basic_plots_path)
        for K in range(1, self.K_range + 1):
            for M in range(1, self.M_range + 1):
                print(f"K={K}, M={M}")
                el_farol = ElFarol(self.N, K, M)
                el_farol.run(self.num_weeks)

                # plot attendance
                if self.plot:
                    el_farol.plot_attendance()

                # plot strategy
                # el_farol.plot_strategy()

                # calculate mean and std of actual attendance
                self.attendance[K - 1][M - 1] = el_farol.attendance
                self.mean_attendance[K - 1][M - 1] = el_farol.mean_attendance
                self.std_attendance[K - 1][M - 1] = el_farol.std_attendance
                # # calculate time to converge
                self.mean_convergence_time[K - 1][M - 1] = el_farol.time_to_converge

        # plot means and convergence time
        if self.plot:
            self.plot_means()
            self.write_results()

    def write_results(self):
        res = [self.mean_attendance, self.std_attendance, self.mean_convergence_time]
        filenames = [f"K {self.K_range}__M{self.M_range}__mean_attendance.csv",
                     f"K {self.K_range}__M{self.M_range}__std_attendance.csv",
                     f"K {self.K_range}__M{self.M_range}__mean_convergence_time.csv"]
        dir_path = os.path.join(data_path, f"K {self.K_range}__M{self.M_range}/")
        os.makedirs(dir_path)
        for j in range(3):
            file_path = os.path.join(dir_path, filenames[j])
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                row_header = ["K = " + str(i + 1) for i in range(self.K_range)]
                col_header = ["M = " + str(i + 1) for i in range(self.M_range)]
                writer.writerow([""] + col_header)
                for i, row in enumerate(res[j]):
                    writer.writerow([row_header[i]] + row.tolist())

    def plot_means(self):
        """
        plot mean actual attendance and convergence time with std and labels
        """
        # plot in dependence with M
        take_care_of_dirs(means_plots_dir + M_dependence_dir, basic_plots_path)
        take_care_of_dirs(converge_time_dir + M_dependence_dir, basic_plots_path)
        for K in range(self.K_range):
            # plot mean actual attendance
            x_axis = range(self.M_range)
            mean_attendance = [self.mean_attendance[K][j] for j in range(self.M_range)]
            y_err = [self.std_attendance[K][j] for j in range(self.M_range)]
            plt.figure()
            plt.errorbar(x_axis, mean_attendance, yerr=y_err, label=f"K={K + 1}", fmt=".")
            for i, val in enumerate(mean_attendance):
                plt.text(x_axis[i], val, f"{int(val) + ((val - int(val)) // 0.1) * 0.1}", ha='center', va='bottom')
            plt.xlabel(r"$M\ [A.U.]$")
            plt.ylabel(r"$Mean Actual Attendance\ [A.U.]$")
            plt.title("Mean Actual Attendance VS M")
            plt.legend()
            plt.savefig(os.path.join(basic_plots_path + means_plots_dir + M_dependence_dir,
                                     f"K {K + 1}__{plot_format}"))
            plt.show()
            plt.close()
            # plot mean convergence time
            mean_convergence_time = [self.mean_convergence_time[K][j] for j in range(self.M_range)]
            plt.figure()
            plt.plot(x_axis, mean_convergence_time, ".", label=f"K={K + 1}")
            for i, val in enumerate(mean_convergence_time):
                plt.text(x_axis[i], val, f"{int(val) + ((val - int(val)) // 0.1) * 0.1}", ha='center', va='bottom')
            plt.xlabel(r"$M\ [A.U.]$")
            plt.ylabel(r"$Mean Time to Converge\ [A.U.]$")
            plt.title("Mean Time to Converge VS M")
            plt.legend()
            plt.savefig(os.path.join(basic_plots_path + converge_time_dir + M_dependence_dir,
                                     f"K {K + 1}__{plot_format}"))
            plt.show()
            plt.close()

        # plot in dependence with K
        take_care_of_dirs(means_plots_dir + K_dependence_dir, basic_plots_path)
        take_care_of_dirs(converge_time_dir + K_dependence_dir, basic_plots_path)
        for M in range(self.M_range):
            # plot mean actual attendance
            x_axis = range(self.K_range)
            mean_attendance = [self.mean_attendance[j][M] for j in range(self.K_range)]
            y_err = [self.std_attendance[j][M] for j in range(self.K_range)]
            plt.figure()
            plt.errorbar(x_axis, mean_attendance, yerr=y_err, label=f"M={M + 1}", fmt=".")
            for i, val in enumerate(mean_attendance):
                plt.text(x_axis[i], val, f"{int(val) + ((val - int(val)) // 0.1) * 0.1}", ha='center', va='bottom')
            plt.xlabel(r"$K\ [A.U.]$")
            plt.ylabel(r"$Mean Actual Attendance\ [A.U.]$")
            plt.title("Mean Actual Attendance VS K")
            plt.legend()
            plt.savefig(os.path.join(basic_plots_path + means_plots_dir + K_dependence_dir,
                                     f"M {M + 1}__{plot_format}"))
            plt.show()
            plt.close()
            # plot mean convergence time
            mean_convergence_time = [self.mean_convergence_time[j][M] for j in range(self.K_range)]
            plt.figure()
            plt.plot(x_axis, mean_convergence_time, ".", label=f"M={M + 1}")
            for i, val in enumerate(mean_convergence_time):
                plt.text(x_axis[i], val, f"{int(val) + ((val - int(val)) // 0.1) * 0.1}", ha='center', va='bottom')
            plt.xlabel(r"$M\ [A.U.]$")
            plt.ylabel(r"$Mean Time to Converge\ [A.U.]$")
            plt.title("Mean Time to Converge VS K")
            plt.legend()
            plt.savefig(os.path.join(basic_plots_path + converge_time_dir + K_dependence_dir,
                                     f"M {M + 1}__{plot_format}"))
            plt.show()
            plt.close()


def take_care_of_dirs(plot_type_dir, plot_path):
    dir_path = os.path.join(plot_path, plot_type_dir)
    if os.path.exists(dir_path):
        for plot in glob.glob(os.path.join(dir_path, plot_format)):
            os.remove(plot)
    else:
        os.makedirs(dir_path)


class ER:
    def __init__(self, K_range, M_range, num_weeks=500, is_run=True, is_extension=True):
        self.K_range = K_range
        self.M_range = M_range
        self.num_weeks = num_weeks
        self.T = 0.6  # np.arange(0.3, 0.1, 0.8)
        self.kupot = np.array([150, 200, 220, 400])
        self.attendance = np.zeros((K_range, M_range, num_weeks + M_range))
        self.mean_attendance = np.zeros((K_range, M_range))
        self.std_attendance = np.zeros((K_range, M_range))
        self.mean_convergence_time = np.zeros((K_range, M_range))
        self.alpha_fit = []
        self.is_extension = is_extension
        if is_run:
            self.run_all()
        else:
            self.get_results()

    def run_all(self):
        for kupa in self.kupot:
            simulation = Simulation(kupa, self.num_weeks, self.K_range, self.M_range, False)
            for K in range(self.K_range):
                for M in range(self.M_range):
                    for i in range(len(simulation.attendance[K][M])):
                        self.attendance[K][M][i] += simulation.attendance[K][M][i]
            self.mean_attendance += simulation.mean_attendance
            self.std_attendance += simulation.std_attendance
            self.mean_convergence_time += simulation.mean_convergence_time
        self.mean_attendance /= len(self.kupot)
        self.std_attendance /= len(self.kupot)
        self.mean_convergence_time /= len(self.kupot)
        self.write_results()
        self.plot_attendance()
        self.plot_means()

    def write_results(self):
        res = [self.mean_attendance, self.std_attendance, self.mean_convergence_time]
        filenames = [f"K {self.K_range}__M{self.M_range}__mean_attendance.csv",
                     f"K {self.K_range}__M{self.M_range}__std_attendance.csv",
                     f"K {self.K_range}__M{self.M_range}__mean_convergence_time.csv"]
        dir_path = os.path.join(data_path, f"K {self.K_range}__M{self.M_range}/")
        os.makedirs(dir_path)
        for j in range(3):
            file_path = os.path.join(dir_path, filenames[j])
            with open(file_path, "w", newline="") as f:
                writer = csv.writer(f)
                row_header = ["K = " + str(i + 1) for i in range(self.K_range)]
                col_header = ["M = " + str(i + 1) for i in range(self.M_range)]
                writer.writerow([""] + col_header)
                for i, row in enumerate(res[j]):
                    writer.writerow([row_header[i]] + row.tolist())

    def get_results(self):
        filenames = [f"K {self.K_range}__M{self.M_range}__mean_attendance.csv",
                     f"K {self.K_range}__M{self.M_range}__std_attendance.csv"]
        data_dir = f"K {self.K_range}__M{self.M_range}_1/"
        df_mean = pd.read_csv(data_path + data_dir + filenames[0])
        df_std = pd.read_csv(data_path + data_dir + filenames[1])
        for K in range(self.K_range):
            for M in range(self.M_range):
                self.mean_attendance[K][M] = float(df_mean[f"M = {M + 1}"][K])
                self.std_attendance[K][M] = float(df_std[f"M = {M + 1}"][K])
        self.plot_means()

    def plot_attendance(self):
        """
        Plots the attendance over time
        """
        take_care_of_dirs(attendance_plots_dir, extension_plots_path)
        for K in range(self.K_range):
            for M in range(self.M_range):
                plt.figure()
                plt.plot(range(len(self.attendance[K][M][:self.M_range + self.num_weeks - M - 1])),
                         self.attendance[K][M][:self.M_range + self.num_weeks - M - 1],
                         label="Attendance Over Days")
                plt.axhline(y=(int(self.T * sum(self.kupot))),
                            color='r', label=f"Threshold from all Kupot Holim={int(self.T * sum(self.kupot))}")
                plt.legend()
                plt.xlabel(r"$Days\ [A.U.]$")
                plt.ylabel(r"$Attendance\ [A.U.]$")
                plt.title(f"ER Attendance VS Weeks, K={K + 1}, M={M + 1}")
                plt.savefig(os.path.join(extension_plots_path + attendance_plots_dir,
                                         f"K {K + 1}__M {M + 1}__{plot_format}"))
                # plt.show()
                plt.close()

    def plot_means(self):
        """
        plot mean actual attendance and convergence time with std and labels
        """
        # plot in dependence with M
        if self.is_extension:
            take_care_of_dirs(means_plots_dir + M_dependence_dir, extension_plots_path)
            take_care_of_dirs(std_size_dir + M_dependence_dir, extension_plots_path)
        else:
            take_care_of_dirs(means_plots_dir + M_dependence_dir, plots_path)
            take_care_of_dirs(std_size_dir + M_dependence_dir, plots_path)
        for K in range(self.K_range):
            # plot mean actual attendance
            x_axis = range(self.M_range)
            mean_attendance = [self.mean_attendance[K][j] for j in range(self.M_range)]
            popt, _ = curve_fit(m_function, x_axis, mean_attendance)
            self.alpha_fit.append(popt[0])
            y_err = [self.std_attendance[K][j] for j in range(self.M_range)]
            plt.figure()
            plt.errorbar(x_axis, mean_attendance, yerr=y_err, label=f"K={K + 1}", fmt=".")
            plt.plot(x_axis, m_function(x_axis, *popt), label=r"fit to $Ae^{-\alpha\ M} + B$")
            for i, val in enumerate(mean_attendance):
                plt.text(x_axis[i], val, f"{int(val) + ((val - int(val)) // 0.1) * 0.1}", ha='center', va='bottom')
            plt.xlabel(r"$M\ [A.U.]$")
            plt.ylabel(r"$Mean Actual Attendance\ [A.U.]$")
            plt.title("Mean Actual Attendance VS M")
            plt.legend()
            if self.is_extension:
                plt.savefig(os.path.join(extension_plots_path + means_plots_dir + M_dependence_dir,
                                         f"K {K + 1}__alpha {popt[0]}__{plot_format}"))
            else:
                plt.savefig(os.path.join(plots_path + means_plots_dir + M_dependence_dir,
                                         f"K {K + 1}__alpha {popt[0]}__{plot_format}"))
            # plt.show()
            plt.close()

            # plot mean convergence time
            std_size = [self.std_attendance[K][j] for j in range(self.M_range)]
            plt.figure()
            plt.plot(x_axis, std_size, ".", label=f"K={K + 1}")
            for i, val in enumerate(std_size):
                plt.text(x_axis[i], val, f"{int(val) + ((val - int(val)) // 0.1) * 0.1}", ha='center', va='bottom')
            plt.xlabel(r"$M\ [A.U.]$")
            plt.ylabel(r"$STD Size\ [A.U.]$")
            plt.title("STD Size VS M")
            plt.legend()
            if self.is_extension:
                plt.savefig(os.path.join(extension_plots_path + std_size_dir + M_dependence_dir,
                                         f"K {K + 1}__{plot_format}"))
            else:
                plt.savefig(os.path.join(plots_path + std_size_dir + M_dependence_dir,
                                         f"K {K + 1}__{plot_format}"))
            plt.show()
            plt.close()

        # plot in dependence with K
        if self.is_extension:
            take_care_of_dirs(means_plots_dir + K_dependence_dir, extension_plots_path)
            take_care_of_dirs(std_size_dir + K_dependence_dir, extension_plots_path)
        else:
            take_care_of_dirs(means_plots_dir + K_dependence_dir, plots_path)
            take_care_of_dirs(std_size_dir + K_dependence_dir, plots_path)
        for M in range(self.M_range):
            # plot mean actual attendance
            x_axis = range(self.K_range)
            mean_attendance = [self.mean_attendance[j][M] for j in range(self.K_range)]
            y_err = [self.std_attendance[j][M] for j in range(self.K_range)]
            plt.figure()
            plt.errorbar(x_axis, mean_attendance, yerr=y_err, label=f"M={M + 1}", fmt=".")
            for i, val in enumerate(mean_attendance):
                plt.text(x_axis[i], val, f"{int(val) + ((val - int(val)) // 0.1) * 0.1}", ha='center', va='bottom')
            plt.xlabel(r"$K\ [A.U.]$")
            plt.ylabel(r"$Mean Actual Attendance\ [A.U.]$")
            plt.title("Mean Actual Attendance VS K")
            plt.legend()
            if self.is_extension:
                plt.savefig(os.path.join(extension_plots_path + means_plots_dir + K_dependence_dir,
                                         f"M {M + 1}__{plot_format}"))
            else:
                plt.savefig(os.path.join(plots_path + means_plots_dir + K_dependence_dir,
                                         f"M {M + 1}__{plot_format}"))
            # plt.show()
            plt.close()
            # plot mean convergence time
            std_size = [self.std_attendance[j][M] for j in range(self.K_range)]
            plt.figure()
            plt.plot(x_axis, std_size, ".", label=f"M={M + 1}")
            for i, val in enumerate(std_size):
                plt.text(x_axis[i], val, f"{int(val) + ((val - int(val)) // 0.1) * 0.1}", ha='center', va='bottom')
            plt.xlabel(r"$M\ [A.U.]$")
            plt.ylabel(r"$STD Size\ [A.U.]$")
            plt.title("STD Size VS M")
            plt.legend()
            if self.is_extension:
                plt.savefig(os.path.join(extension_plots_path + std_size_dir + K_dependence_dir,
                                         f"M {M + 1}__{plot_format}"))
            else:
                plt.savefig(os.path.join(plots_path + std_size_dir + K_dependence_dir,
                                         f"M {M + 1}__{plot_format}"))
            plt.show()
            plt.close()


plots_path = "C:/Users/USER-1/OneDrive - huji.ac.il/Documents/Third Year/Using Exact Sciences Modeling " \
             "Tools to Understand social phenomenas/plots/"
extension_plots_path = plots_path + "extension/"
basic_plots_path = plots_path + "basic/"
data_path = "C:/Users/USER-1/OneDrive - huji.ac.il/Documents/Third Year/Using Exact Sciences Modeling " \
            "Tools to Understand social phenomenas/data/"
attendance_plots_dir = "attendance plots/"
means_plots_dir = "mean plots/"
converge_time_dir = "converge plots/"
std_size_dir = "std_size_plots/"
K_dependence_dir = "K dependence/"
M_dependence_dir = "M dependence/"
plot_format = ".png"


def m_function(m, alpha, a, c):
    return a * (1 - np.exp(- alpha * m)) + c


if __name__ == "__main__":
    # sim = Simulation(200, 500, 20, 20)
    # sim1 = ER(15, 15)
    plots_extension = ER(15, 15, is_run=False)
    plots_basic = ER(20, 20, is_run=False, is_extension=False)
