import pandas as pd
import numpy as np
import os

CHART_PATH = os.path.join(os.path.dirname(__file__), '../color_chart.csv')

class ColorChart:
    def __init__(self, chart_path=CHART_PATH):
        self.chart_path = chart_path
        self.df = pd.read_csv(chart_path)
        self.colors = self.df[['Red', 'Green', 'Blue']].values.astype(int)
        self.concentrations = self.df['Concentration_mg_L'].values.astype(float)
        self.hexes = self.df['Hex'].values

    def find_closest(self, rgb):
        rgb = np.array(rgb).reshape(1, 3)
        dists = np.linalg.norm(self.colors - rgb, axis=1)
        idx = np.argmin(dists)
        return {
            'concentration': self.concentrations[idx],
            'hex': self.hexes[idx],
            'rgb': tuple(self.colors[idx]),
            'distance': dists[idx]
        }

    def add_entry(self, concentration, hex_code, r, g, b):
        new_row = pd.DataFrame({
            'Concentration_mg_L': [concentration],
            'Hex': [hex_code],
            'Red': [r],
            'Green': [g],
            'Blue': [b]
        })
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.df.to_csv(self.chart_path, index=False)
        self.colors = self.df[['Red', 'Green', 'Blue']].values.astype(int)
        self.concentrations = self.df['Concentration_mg_L'].values.astype(float)
        self.hexes = self.df['Hex'].values 