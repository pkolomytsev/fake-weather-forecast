"""
A script to demonstrate a “naive” approach to predicting monthly average temperatures using long-term observational
data and various methods of extrapolating values.

Usage:
    # analyse all data sets with increasing the experimental range to 50 years and forecast range to 5 years
    python show-forecast.py -e 50 -f 5

    # analyse data sets and render forecast graphics for the selected stations (see ./data or data.zip)
    python show-forecast.py -e 10 -f 1 22003 28009 29467
"""

__author__ = 'pkolomytsev'


import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Dict, Tuple, List, Union
import zipfile

import matplotlib.pyplot as pyplot
import numpy
import scipy.optimize as optimize


ZERO_YEAR = 1970


@dataclass
class Options:
    """Application options class."""

    DEFAULT_EXPERIMENTAL_RANGE = 25 * 12  # 25 years
    DEFAULT_FORECAST_RANGE = 1 * 12  # 1 year
    DEFAULT_DATA_ROOT = str(Path(__file__).parent / 'data')
    DEFAULT_DATA_PACK = str(Path(__file__).parent / 'data.zip')

    experimental_range: int = DEFAULT_EXPERIMENTAL_RANGE
    forecast_range: int = DEFAULT_FORECAST_RANGE
    data_root: str = DEFAULT_DATA_ROOT
    data_pack: str = DEFAULT_DATA_PACK
    stations: List[int] = field(default_factory=list)
    profile: bool = False

    @classmethod
    def create_argument_parser(cls) -> argparse.ArgumentParser:
        """Create a parser for the text UI.

        :return: ArgumentParser object
        """

        def range_multiplier(years: str) -> int:
            """Convert the `year` value from the command line to the number of
            months.

            :param years: years value
            :return: months value
            """
            return int(years) * 12

        parser = argparse.ArgumentParser(description='Fake weather forecast')
        parser.add_argument(
            'stations',
            nargs='*',
            type=int,
            help='List of stations ID to render forecast.'
        )
        parser.add_argument(
            '-e', '--experimental-range',
            type=range_multiplier,
            default=cls.DEFAULT_EXPERIMENTAL_RANGE,
            metavar='YEARS',
            help='Experimental range, which includes training and forecast ranges.'
        )
        parser.add_argument(
            '-f', '--forecast-range',
            type=range_multiplier,
            default=cls.DEFAULT_FORECAST_RANGE,
            metavar='YEARS',
            help='Forecast range (last N years).'
        )
        parser.add_argument(
            '-p', '--profile',
            action='store_true',
            help='Profile predictors.'
        )
        parser.add_argument(
            '--data-root',
            default=cls.DEFAULT_DATA_ROOT,
            help='Data root path.'
        )
        parser.add_argument(
            '--data-pack',
            default=cls.DEFAULT_DATA_PACK,
            help='Data archive path.'
        )
        return parser

    @classmethod
    def from_command_line(cls, options: List[str] = None) -> 'Options':
        """Create ``Options`` from command line parameters or from a list of
        parameters.

        :param options: list of parameters
        :return: Options object
        """
        argument_parser = cls.create_argument_parser()
        args = argument_parser.parse_args(args=options)
        return cls(**vars(args))


@dataclass
class DataSet:
    """Temperature data and their corresponding time series."""

    dates: numpy.ndarray
    values: numpy.ndarray

    def __getitem__(self, item) -> Union['DataSet', Tuple[int, float]]:
        if isinstance(item, slice):
            return type(self)(self.dates.__getitem__(item), self.values.__getitem__(item))
        else:
            return self.dates[item], self.values[item]

    def __len__(self):
        return len(self.dates)

    @staticmethod
    def date_to_stamp(date: int) -> str:
        """Convert the `date` value to a string timestamp.

        :param date: value from time series
        :return: timestamp
        """
        year = date // 12 + ZERO_YEAR
        month = date % 12 + 1
        return f'{year}.{month:02d}'

    @classmethod
    def from_file(cls, path: str) -> 'DataSet':
        """Load temperature data from a file.

        :param path: path to the file
        :return: DataSet object
        """
        dates = []
        t_values = []
        with open(path, encoding='utf-8') as f:
            data_reader = csv.reader(f, delimiter=';')
            for station_id, year, *values in data_reader:
                for month, t in enumerate(values):
                    try:
                        date = (int(year) - ZERO_YEAR) * 12 + month
                        t = float(t)
                    except ValueError:
                        continue
                    dates.append(date)
                    t_values.append(t)
        return cls(numpy.array(dates), numpy.array(t_values))


class DataVault:
    """Experimental data storage manager."""

    STATIONS_TABLE = 'stations.txt'

    def __init__(self, root: str = Options.DEFAULT_DATA_ROOT, pack: str = Options.DEFAULT_DATA_PACK):
        """Unpack experimental data (if need) and create a weather station
        catalog.

        :param root: path to data root folder
        :param pack: path to experimental data archive
        """
        self.root = Path(root)
        self.pack = Path(pack)

        if not self.root.is_dir():
            self.root.mkdir()
            with zipfile.ZipFile(self.pack) as archive:
                archive.extractall(self.root)

        self.stations: Dict[int, str] = {}
        with (self.root / self.STATIONS_TABLE).open(encoding='utf-8') as f:
            for line in f:
                sid, name = line.split(' ', 1)
                self.stations[int(sid)] = name.strip()

        self.cache: Dict[int, DataSet] = {}

    def get_climate_data(self, station_id: int) -> DataSet:
        """Get climate data for the selected weather station.

        :param station_id: weather station ID
        :return: DataSet object
        """
        if station_id not in self.cache:
            path = self.root / f'{station_id}.txt'
            self.cache[station_id] = DataSet.from_file(str(path))
        return self.cache[station_id]

    def iter_all_climate_data(self) -> Iterable[Tuple[int, DataSet]]:
        """Iter DataSets for all available weather station.

        :return: generator of pairs [station ID, DataSet]
        """
        for file in self.root.iterdir():
            if file.name == self.STATIONS_TABLE:
                continue
            sid = int(os.path.splitext(file.name)[0])
            yield sid, self.get_climate_data(sid)


def make_prediction(ext_method: Callable, dates: numpy.ndarray) -> DataSet:
    """Apply the extrapolation method to the time series.

    :param ext_method: extrapolation method
    :param dates: time series
    :return: new DataSet with weather prediction
    """
    prediction = ext_method(dates)
    return DataSet(numpy.array(dates), prediction)


def eval_error(experimental_values: numpy.ndarray, prediction_values: numpy.ndarray) -> float:
    """Evaluate absolute error value using root-sum-square uncertainty.

    :param experimental_values: known temperatures
    :param prediction_values: predicted temperatures
    :return: error value
    """
    return math.sqrt(numpy.sum((experimental_values - prediction_values) ** 2) / len(experimental_values))


def get_extrapolation_function_by_least_squares(data: DataSet) -> Callable:
    """Train a least squares model.

    :param data: training range
    :return: extrapolation function (t = f(date))
    """
    period = numpy.pi / 6

    def fit(date, a, b, c, d, e):
        return numpy.cos((date * a + b) * period) * c + date * d + e

    opt, cov = optimize.curve_fit(fit, data.dates, data.values)
    return lambda date: fit(date, *opt)


def get_simple_extrapolation_function(data: DataSet, stat_method: Callable) -> Callable:
    """Train a simple model using average or median temperatures.

    :param data: training range
    :param stat_method: numpy.median or numpy.average
    :return: extrapolation function (t = f(date))
    """
    values_by_months = defaultdict(list)
    for d, t in zip(data.dates, data.values):
        month = d % 12
        values_by_months[month].append(t)
    predictions_by_months = {}
    for month in values_by_months:
        predictions_by_months[month] = stat_method(values_by_months[month])

    def fit(date):
        if isinstance(date, Iterable):
            return numpy.array([predictions_by_months[u % 12] for u in date])
        else:
            return predictions_by_months[int(date) % 12]
    return fit


def render_prediction(experiment: DataSet, prediction: DataSet, station: str, method: str = None) -> None:
    """Render a forecast graphic (PNG).

    :param experiment: DataSet with known temperatures
    :param prediction: DataSet with predicted temperatures
    :param station: weather station name
    :param method: extrapolation method name
    :return: none
    """
    if len(experiment.dates) == 12:
        xticks = experiment.dates
    else:
        xticks = experiment.dates[::12]
    figure, axis = pyplot.subplots()
    figure.set_size_inches(16, 9)

    if method:
        legend = ('Experimental', f'Fit by {method}')
        fig_type = 'forecast-by-' + method.replace(' ', '-')
    else:
        legend = ('Experimental', 'Fit')
        fig_type = 'forecast'

    pyplot.plot(experiment.dates, experiment.values, 'vk:')
    pyplot.plot(prediction.dates, prediction.values, '^r:')

    axis.set_xlabel('Date', fontsize=14)
    axis.set_ylabel(r'Average Monthly Temperature ($^\circ$C)', fontsize=14)
    pyplot.title(f'Climate data ({station})', fontsize=18)
    pyplot.xticks(xticks, (DataSet.date_to_stamp(d) for d in xticks), rotation=70)
    pyplot.grid()
    pyplot.legend(legend, loc='upper left', fontsize=14)
    pyplot.savefig(f'{station}-{fig_type}.png', dpi=300)


def render_profile(profile: Dict[str, List[float]], profile_ticks: List[int], forecast_depth: int) -> None:
    """Render profiling results.

    :param profile: profile data for each extrapolation method
    :param profile_ticks: profiling range in years
    :param forecast_depth: forecast depth for the given profile
    :return: none
    """
    legend = sorted(profile)
    figure, axis = pyplot.subplots()
    for method in legend:
        pyplot.plot(profile_ticks, profile[method], linewidth=3)
    figure.set_size_inches(16, 9)
    axis.set_xlabel('Experiment depth (years)', fontsize=14)
    axis.set_ylabel('Predictor score (%)', fontsize=14)
    pyplot.title(f'Predictors profile ({forecast_depth} years)', fontsize=18)
    pyplot.grid()
    pyplot.legend(legend, loc='upper left', fontsize=14)
    pyplot.savefig(f'predictors-profile-{forecast_depth:02d}-years.png', dpi=300)


class App:
    """Application class."""

    STAT_T = Dict[int, Dict[str, float]]

    EXT_METHODS = {
        'least squares': get_extrapolation_function_by_least_squares,
        'median': lambda d: get_simple_extrapolation_function(d, numpy.median),
        'average': lambda d: get_simple_extrapolation_function(d, numpy.average)
    }

    def __init__(self, options: Options = None):
        """Create a DataVault using the given options or using a default one.

        :param options: Options object
        """
        if not options:
            options = Options()
        self.options = options
        self.vault = DataVault(options.data_root, options.data_pack)

    def show(self, station_id: int) -> None:
        """Create and render forecast for the selected weather station using
        all extrapolation methods.

        :param station_id: weather station ID
        :return: none
        """
        station_name = self.vault.stations[station_id]
        print(f'{station_id:05d}: {station_name}')
        data = self.vault.get_climate_data(station_id)
        training_data = data[-self.options.experimental_range:-self.options.forecast_range]
        forecast_data = data[-self.options.forecast_range:]

        for method, builder in self.EXT_METHODS.items():
            f = builder(training_data)
            prediction = make_prediction(f, forecast_data.dates)
            error = eval_error(forecast_data.values, prediction.values)
            print(f'{method} fit error = {error:3.3f}')
            render_prediction(forecast_data, prediction, station_name, method)

    def analyse_all_data_sets(self, experimental_range: int = None, forecast_range: int = None) -> STAT_T:
        """Analyse all available DataSets using all extrapolation methods.

        :param experimental_range: range, which includes training and forecast ranges
        :param forecast_range: right side of the experimental range
        :return: prediction error statistics
        """
        if not experimental_range:
            experimental_range = self.options.experimental_range
        if not forecast_range:
            forecast_range = self.options.forecast_range
        stat = {}

        for sid, data in self.vault.iter_all_climate_data():
            if len(data) < experimental_range:
                continue
            training_data = data[-experimental_range:-forecast_range]
            forecast_data = data[-forecast_range:]
            station_stat = stat[sid] = {}

            for method, builder in self.EXT_METHODS.items():
                f = builder(training_data)
                prediction = make_prediction(f, forecast_data.dates)
                station_stat[method] = eval_error(forecast_data.values, prediction.values)
        return stat

    def print_predictors_stat(self, stat: STAT_T) -> None:
        """Print the best and the worst forecast results.

        :param stat: prediction error statistics
        :return: none
        """
        stations = self.vault.stations

        print(f'Predictors score:')
        asb_score, rel_score = self.get_methods_score(stat)
        for method in sorted(asb_score, key=lambda u: asb_score[u], reverse=True):
            print(f'  {method}: {rel_score[method]:.1f}% ({asb_score[method]})')
        print()

        def print_sorted_stat(_stat: List, _method: str) -> None:
            print(f'Worst {_method} predictions')
            for _sid, _errors in _stat[-5:]:
                print(f'  {_sid:05d}: {stations[_sid]:20s} -- {_errors[_method]:6.3f}')
            print()
            print(f'Best {_method} predictions')
            for _sid, _errors in _stat[:5]:
                print(f'  {_sid:05d}: {stations[_sid]:20s} -- {_errors[_method]:6.3f}')
            print()

        for method in asb_score:
            sorted_stat = sorted(stat.items(), key=lambda u: u[1][method])
            print_sorted_stat(sorted_stat, method)

    def profile_methods(self) -> None:
        """Create and render a profile for all extrapolation methods using
        `analyse_all_data_sets` method and various experimental ranges.

        :return: none
        """
        experimental_range_list = list(range(120, self.options.experimental_range + 1, 60))
        forecast_range_list = list(range(12, self.options.forecast_range + 1, 24))
        for forecast_range in forecast_range_list:
            profile = defaultdict(list)
            profile_ticks = []
            for experimental_range in experimental_range_list:
                experiment_depth = experimental_range // 12
                print(f'profile: forecast - {forecast_range // 12}; experiment - {experiment_depth}')
                stat = self.analyse_all_data_sets(experimental_range, forecast_range)
                rel_score = self.get_methods_score(stat)[1]
                for method, value in rel_score.items():
                    profile[method].append(value)
                profile_ticks.append(experiment_depth)
            render_profile(profile, profile_ticks, forecast_range // 12)

    @staticmethod
    def get_methods_score(stat: STAT_T) -> Tuple[Dict[str, int], Dict[str, float]]:
        """Calculate an absolute and relative predictors score.

        :param stat: prediction error statistics
        :return: pair of [absolute score, relative score]
        """
        abs_score = defaultdict(int)
        for errors in stat.values():
            winner = min(errors, key=lambda u: errors[u])
            abs_score[winner] += 1
        total = sum(abs_score.values())
        rel_score = {method: value / total * 100 for method, value in abs_score.items()}
        return abs_score, rel_score

    @classmethod
    def main(cls) -> int:
        """Application entry point.

        :return: 0 or 1 in case of any error
        """
        options = Options.from_command_line()
        app = cls(options)
        if options.stations:
            for sid in options.stations:
                app.show(sid)
        elif options.profile:
            app.profile_methods()
        else:
            stat = app.analyse_all_data_sets()
            app.print_predictors_stat(stat)
        return 0


if __name__ == '__main__':
    exit(App.main())
