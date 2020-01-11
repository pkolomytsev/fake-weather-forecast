import argparse
from collections import defaultdict
import csv
from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Callable, Iterable, Dict, Tuple, List
import zipfile

import matplotlib.pyplot as pyplot
import numpy
import scipy.optimize as optimize


ZERO_YEAR = 1970


@dataclass
class Options:
    DEFAULT_EXPERIMENTAL_RANGE = 25 * 12  # 25 years
    DEFAULT_FORECAST_RANGE = 1 * 12  # 1 year
    DEFAULT_DATA_ROOT = Path(__file__).parent / 'data'
    DEFAULT_DATA_PACK = Path(__file__).parent / 'data.zip'

    experimental_range: int = DEFAULT_EXPERIMENTAL_RANGE
    forecast_range: int = DEFAULT_FORECAST_RANGE
    data_root: Path = DEFAULT_DATA_ROOT
    data_pack: Path = DEFAULT_DATA_PACK
    stations: List[int] = field(default_factory=list)
    profile: bool = False

    @classmethod
    def create_argument_parser(cls) -> argparse.ArgumentParser:
        def range_multiplier(years: str) -> int:
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
            type=Path,
            default=cls.DEFAULT_DATA_ROOT,
            help='Data root path.'
        )
        parser.add_argument(
            '--data-pack',
            type=Path,
            default=cls.DEFAULT_DATA_PACK,
            help='Data archive path.'
        )
        return parser

    @classmethod
    def from_command_line(cls, options: List[str] = None) -> 'Options':
        argument_parser = cls.create_argument_parser()
        args = argument_parser.parse_args(args=options)
        return cls(**vars(args))


@dataclass
class DataSet:
    dates: numpy.ndarray
    values: numpy.ndarray

    def __getitem__(self, item) -> 'DataSet':
        if isinstance(item, slice):
            return type(self)(self.dates.__getitem__(item), self.values.__getitem__(item))
        else:
            return type(self)(numpy.array([self.dates[item]]), numpy.array([self.values[item]]))

    @staticmethod
    def date_to_stamp(date: int) -> str:
        year = date // 12 + ZERO_YEAR
        month = date % 12 + 1
        return f'{year}.{month:02d}'

    @classmethod
    def from_file(cls, path: str) -> 'DataSet':
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
    STATIONS_TABLE = 'stations.txt'

    def __init__(self, root: Path = Options.DEFAULT_DATA_ROOT, pack: Path = Options.DEFAULT_DATA_PACK):
        self.root = root
        self.pack = pack

        if not self.root.is_dir():
            self.root.mkdir()
            with zipfile.ZipFile(self.pack) as archive:
                archive.extractall(self.root)

        self.stations: Dict[int, str] = {}
        with (self.root / self.STATIONS_TABLE).open(encoding='utf-8') as f:
            for line in f:
                sid, name = line.split(' ', 1)
                self.stations[int(sid)] = name.strip()

    def get_climate_data(self, station_id: int) -> DataSet:
        path = self.root / f'{station_id}.txt'
        return DataSet.from_file(str(path))

    def iter_all_climate_data(self) -> Iterable[Tuple[int, DataSet]]:
        for file in self.root.iterdir():
            if file.name == self.STATIONS_TABLE:
                continue
            sid = int(os.path.splitext(file.name)[0])
            yield sid, self.get_climate_data(sid)


def make_prediction(ext_method: Callable, dates: numpy.ndarray) -> DataSet:
    prediction = ext_method(dates)
    return DataSet(numpy.array(dates), prediction)


def eval_error(experimental_values: numpy.ndarray, prediction_values: numpy.ndarray) -> float:
    error = 0.0
    for exp, pred in zip(experimental_values, prediction_values):
        error += (exp - pred) ** 2
    error = math.sqrt(error / len(experimental_values))
    return error


def get_extrapolation_function_by_least_squares(data: DataSet) -> Callable:
    period = numpy.pi / 6

    def fit(date, a, b, c, d, e):
        return numpy.cos((date * a + b) * period) * c + date * d + e

    opt, cov = optimize.curve_fit(fit, data.dates, data.values)
    return lambda date: fit(date, *opt)


def get_simple_extrapolation_function(data: DataSet, stat_method: Callable) -> Callable:
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


class App:
    STAT_T = Dict[int, Dict[str, float]]

    def __init__(self, options: Options = None):
        if not options:
            options = Options()
        self.options = options
        self.vault = DataVault(options.data_root, options.data_pack)

    def show(self, station_id: int) -> None:
        station_name = self.vault.stations[station_id]
        print(f'{station_id:05d}: {station_name}')
        data = self.vault.get_climate_data(station_id)
        training_data = data[-self.options.experimental_range:-self.options.forecast_range]
        forecast_data = data[-self.options.forecast_range:]

        f = get_extrapolation_function_by_least_squares(training_data)
        prediction = make_prediction(f, forecast_data.dates)
        error = eval_error(forecast_data.values, prediction.values)
        print(f'Least squares fit error = {error:3.3f}')
        self.render_prediction(forecast_data, prediction, station_name, 'least squares')

        f = get_simple_extrapolation_function(training_data, numpy.median)
        prediction = make_prediction(f, forecast_data.dates)
        error = eval_error(forecast_data.values, prediction.values)
        print(f'Median fit error = {error:3.3f}')
        self.render_prediction(forecast_data, prediction, station_name, 'median')

        f = get_simple_extrapolation_function(training_data, numpy.average)
        prediction = make_prediction(f, forecast_data.dates)
        error = eval_error(forecast_data.values, prediction.values)
        print(f'Average fit error = {error:3.3f}')
        self.render_prediction(forecast_data, prediction, station_name, 'average')

    @staticmethod
    def render_prediction(experiment: DataSet, prediction: DataSet, station_name: str, method: str = None) -> None:
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
        pyplot.title(f'Climate data ({station_name})')
        pyplot.xticks(xticks, (DataSet.date_to_stamp(d) for d in xticks), rotation=70)
        pyplot.grid()
        pyplot.legend(legend, loc='upper left')
        pyplot.savefig(f'{station_name}-{fig_type}.png', dpi=300)

    def analyse_all_data_sets(self) -> STAT_T:
        stat = {}

        builders_by_methods = {
            'least squares': get_extrapolation_function_by_least_squares,
            'median': lambda d: get_simple_extrapolation_function(d, numpy.median),
            'average': lambda d: get_simple_extrapolation_function(d, numpy.average)
        }

        for sid, data in self.vault.iter_all_climate_data():
            training_data = data[-self.options.experimental_range:-self.options.forecast_range]
            forecast_data = data[-self.options.forecast_range:]
            station_stat = stat[sid] = {}

            for method, builder in builders_by_methods.items():
                f = builder(training_data)
                prediction = make_prediction(f, forecast_data.dates)
                station_stat[method] = eval_error(forecast_data.values, prediction.values)
        return stat

    @staticmethod
    def get_methods_score(stat: STAT_T) -> Dict[str, int]:
        score = defaultdict(int)
        for errors in stat.values():
            winner = min(errors, key=lambda u: errors[u])
            score[winner] += 1
        return score

    def print_predictors_stat(self, stat: STAT_T) -> None:
        stations = self.vault.stations

        print(f'Predictors score:')
        score = self.get_methods_score(stat)
        total = sum(score.values())
        for method, value in sorted(score.items(), key=lambda u: u[1], reverse=True):
            print(f'  {method}: {value / total * 100:.1f}% ({value})')
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

        for method in score:
            sorted_stat = sorted(stat.items(), key=lambda u: u[1][method])
            print_sorted_stat(sorted_stat, method)

    def profile_methods(self):
        for forecast_range in (1, 3, 5):
            profile = defaultdict(list)
            for experimental_range in range(5, 70, 5):
                stat = self.analyse_all_data_sets(experimental_range * 12, forecast_range * 12)
                score = self.get_methods_score(stat)
                total = sum(score.values())
                for method, value in score:
                    profile[method].append(value / total)

    def render_profile(self, profile, forecast_range):
        pass

    @classmethod
    def main(cls) -> int:
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
