import numpy as np
from panqec.usage import TextPlotter


class TestTextPlotter:

    def test_plot_lissajous_figure(self):
        plotter = TextPlotter(width=79, height=34)
        x = np.linspace(-10, 10, 1001)
        plotter.plot(np.cos(3*x), np.sin(3*x), '.')
        plotter.plot(0.5*np.cos(3*x), 0.5*np.sin(3*x), '@')
        plotter.plot(np.cos(3*x) - np.sin(2*x), np.cos(3*x) + np.sin(2*x), 'o')
        plotter.xlabel('x (m)')
        plotter.ylabel('y (m)')
        plotter.title('Cool title goes here')
        plotter.show()
