# fake-weather-forecast

## Description

Demonstration of a "naive" approach to predicting monthly average temperatures using data from long-term observations 
and various methods of extrapolating values.

Used models:
* monthly average temperature model;
* monthly median temperature model;
* periodic model with a trend line by the method of least squares.

Sources:
* [База данных ВНИИГМИ-МЦД](http://meteo.ru/data/162-temperature-precipitation)

## Installation

This project uses [Python 3.6+](https://www.python.org/downloads/release/python-368/). Dependency installation:

    pip install -r requirements.txt

## Usage

    # print help 
    python show-forecast.py --help

    # analyse all data sets
    python show-forecast.py

    # analyse all data sets with increasing the experimental range to 50 years and forecast range to 5 years
    python show-forecast.py -e 50 -f 5

    # analyse data sets and render forcast graphics for the selected stations (see ./data or data.zip)
    python show-forecast.py -e 10 -f 1 22003 28009 29467
