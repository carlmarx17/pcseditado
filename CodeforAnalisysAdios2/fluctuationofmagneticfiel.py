#!/usr/bin/env python3
"""
fluctuationofmagneticfiel.py
============================
Visualización de estructuras magnéticas (fluctuaciones individuales).
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import re
import imageio
from pathlib import Path
from scipy.ndimage import gaussian_filter
from matplotlib.colors import LinearSegmentedColormap

from data_reader import PICDataReader

plt.switch_backend('Agg')

class FieldImagePlotter:
    def __init__(self, em_pattern='pfd.*.bp', B0=0.01, fluct_amp=0.1, outdir='field_images', 
                 smooth_sigma=0.8, dyn_scale=True, comp_magnitude=True, comps_to_plot=['Bx', 'By', 'Bz'],
                 fixed_scale=True):
        self.em_pattern = em_pattern
        self.B0 = B0
        self.fluct_amp = fluct_amp
        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True, parents=True)
        self.smooth_sigma = smooth_sigma
        self.dyn_scale = dyn_scale
        self.comp_magnitude = comp_magnitude
        self.comps_to_plot = comps_to_plot
        self.fixed_scale = fixed_scale
        self.global_vmin = {}
        self.global_vmax = {}
        self.file_map = PICDataReader.find_files(self.em_pattern)
        
        # Paleta de colores personalizada - Alto contraste
        self.palettes = {
            'Bx': self.create_dark_divergent_cmap('blue', 'red'),
            'By': self.create_dark_divergent_cmap('green', 'purple'),
            'Bz': self.create_dark_divergent_cmap('cyan', 'orange'),
            'Bmag': self.create_yellow_green_cmap()  # Paleta amarillo-verde-negro
        }

    def create_dark_divergent_cmap(self, color1, color2):
        """Mapas divergentes con alto contraste y extremos oscuros"""
        palettes = {
            'blue_red': ['#00001F', '#00008F', '#0000FF', '#0066FF', '#FF0000', '#8B0000', '#450000'],
            'green_purple': ['#001F00', '#004400', '#00FF00', '#66FF00', '#800080', '#4B0082', '#2A0045'],
            'cyan_orange': ['#002A2A', '#006666', '#00FFFF', '#00CCCC', '#FF8C00', '#FF4500', '#8B2500']
        }
        
        if 'blue' in color1 and 'red' in color2:
            colors = palettes['blue_red']
        elif 'green' in color1 and 'purple' in color2:
            colors = palettes['green_purple']
        else:
            colors = palettes['cyan_orange']
            
        return LinearSegmentedColormap.from_list(f'hi_contrast_{color1}_{color2}', colors, N=256)

    def create_yellow_green_cmap(self):
        """Paleta secuencial de alto contraste (amarillo-verde-negro)"""
        colors = [
            '#FFFF00', '#EEEE00', '#CCCC00', '#AAAA00', '#888800',
            '#00FF00', '#008800', '#004400', '#002200', '#001100', '#000000'
        ]
        return LinearSegmentedColormap.from_list('hi_contrast_yellow_green', colors, N=256)

    def _process_component(self, data):
        """Procesa un componente: normalización y suavizado"""
        fluct = (data - np.mean(data)) / self.B0
        if self.smooth_sigma > 0:
            fluct = gaussian_filter(fluct, sigma=self.smooth_sigma)
        return fluct

    def _get_slice(self, bx3d, by3d, bz3d, plane, slice_index):
        # The data shape from PSC is (Nz, Ny, Nx). 
        # For a 2D Y-Z simulation, Nx=1, Ny>1, Nz>1, shape is e.g. (512, 384, 1).
        shape = bx3d.shape
        
        if plane == 'xy':
            k = slice_index if slice_index is not None else shape[0] // 2
            return bx3d[k, :, :].squeeze(), by3d[k, :, :].squeeze(), bz3d[k, :, :].squeeze(), 'x', 'y'
        elif plane == 'xz':
            k = slice_index if slice_index is not None else shape[1] // 2
            return bx3d[:, k, :].squeeze(), by3d[:, k, :].squeeze(), bz3d[:, k, :].squeeze(), 'x', 'z'
        elif plane == 'yz':
            k = slice_index if slice_index is not None else shape[2] // 2
            # Here data is (Nz, Ny). After squeeze we have (Nz, Ny)
            # which we later transpose with .T, resulting in (Ny, Nz).
            # So X-axis is Z, Y-axis is Y.
            return bx3d[:, :, k].squeeze(), by3d[:, :, k].squeeze(), bz3d[:, :, k].squeeze(), 'z', 'y'
        else:
            raise ValueError("Plano debe ser 'xy', 'xz' o 'yz'")

    def compute_global_scales(self, steps, plane, slice_index=None):
        """Calcula escalas globales para mantener consistencia en colores"""
        print("Calculando escalas globales para colores consistentes...")
        global_extremas = {comp: {'min': [], 'max': []} for comp in ['Bx', 'By', 'Bz', 'Bmag']}
        
        for step in steps:
            fn = self.file_map.get(step)
            if not fn: continue
            
            try:
                fields = PICDataReader.read_multiple_fields_3d(fn, "jeh-", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
                bx3d = fields["hx_fc/p0/3d"] * self.B0
                by3d = fields["hy_fc/p0/3d"] * self.B0
                bz3d = fields["hz_fc/p0/3d"] * self.B0
            except Exception as e:
                print(f"Error leyendo step {step} para limites: {e}")
                continue
                
            bx2d, by2d, bz2d, _, _ = self._get_slice(bx3d, by3d, bz3d, plane, slice_index)
            
            if self.comp_magnitude:
                mag = np.sqrt((bx2d - np.mean(bx2d))**2 + (by2d - np.mean(by2d))**2 + (bz2d - np.mean(bz2d))**2) / self.B0
                if self.smooth_sigma > 0:
                    mag = gaussian_filter(mag, sigma=self.smooth_sigma)
                global_extremas['Bmag']['min'].append(np.min(mag))
                global_extremas['Bmag']['max'].append(np.max(mag))

            for comp, data in [('Bx', bx2d), ('By', by2d), ('Bz', bz2d)]:
                if comp in self.comps_to_plot:
                    fluct = self._process_component(data)
                    global_extremas[comp]['min'].append(np.min(fluct))
                    global_extremas[comp]['max'].append(np.max(fluct))
                    
        for comp in global_extremas:
            if global_extremas[comp]['min']:
                self.global_vmin[comp] = np.percentile(global_extremas[comp]['min'], 1)
                self.global_vmax[comp] = np.percentile(global_extremas[comp]['max'], 99)
                if comp != 'Bmag':
                    abs_max = max(abs(self.global_vmin[comp]), abs(self.global_vmax[comp]))
                    self.global_vmin[comp] = -abs_max
                    self.global_vmax[comp] = abs_max
                    
        print("Escalas globales calculadas:")
        for comp in self.global_vmin:
            print(f"  {comp}: [{self.global_vmin[comp]:.4f}, {self.global_vmax[comp]:.4f}]")

    def plot_snapshot(self, step, plane='xy', slice_index=None):
        fn = self.file_map.get(step)
        if fn is None:
            return False
            
        try:
            fields = PICDataReader.read_multiple_fields_3d(fn, "jeh-", ["hx_fc/p0/3d", "hy_fc/p0/3d", "hz_fc/p0/3d"])
            bx3d = fields["hx_fc/p0/3d"] * self.B0
            by3d = fields["hy_fc/p0/3d"] * self.B0
            bz3d = fields["hz_fc/p0/3d"] * self.B0
        except Exception as e:
            print(f"Error procesando step {step}: {e}")
            return False

        bx2d, by2d, bz2d, xlabel, ylabel = self._get_slice(bx3d, by3d, bz3d, plane, slice_index)

        time_omega_ci = step * 0.00104961

        # Magnitud
        if self.comp_magnitude:
            mag = np.sqrt((bx2d - np.mean(bx2d))**2 + (by2d - np.mean(by2d))**2 + (bz2d - np.mean(bz2d))**2) / self.B0
            if self.smooth_sigma > 0:
                mag = gaussian_filter(mag, sigma=self.smooth_sigma)
            
            if self.fixed_scale and 'Bmag' in self.global_vmin:
                vmin, vmax = self.global_vmin['Bmag'], self.global_vmax['Bmag']
            elif self.dyn_scale:
                vmax = max(np.percentile(mag, 99.8), 1e-4) # Mejor contraste 99.8 
                vmin = 0
            else:
                vmin, vmax = 0, self.fluct_amp

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(mag.T, origin='lower', cmap=self.palettes['Bmag'], vmin=vmin, vmax=vmax, aspect='auto')
            fig.colorbar(im, ax=ax).set_label(r'$|\delta B| / B_0$', fontsize=14)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(rf'Magnitude $|\delta B| / B_0$ | Step {step} | $t \approx {time_omega_ci:.2f} \Omega_{{ci}}^{{-1}}$', fontsize=14)
            ax.grid(True, linestyle=':', alpha=0.7)
            
            outname = self.outdir / f'Bmag_fluct_step{step}_{plane}.png'
            fig.savefig(outname, dpi=200, bbox_inches='tight')
            plt.close(fig)

        # Componentes
        for comp, data in [('Bx', bx2d), ('By', by2d), ('Bz', bz2d)]:
            if comp not in self.comps_to_plot: continue
                
            fluct = self._process_component(data)
            
            if self.fixed_scale and comp in self.global_vmin:
                vmin, vmax = self.global_vmin[comp], self.global_vmax[comp]
            elif self.dyn_scale:
                vmax = max(np.percentile(np.abs(fluct), 99.8), 1e-4)
                vmin = -vmax
            else:
                vmin, vmax = -self.fluct_amp, self.fluct_amp

            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(fluct.T, origin='lower', cmap=self.palettes[comp], vmin=vmin, vmax=vmax, aspect='auto')
            fig.colorbar(im, ax=ax).set_label(rf'$\delta {comp} / B_0$', fontsize=14)
            ax.set_xlabel(xlabel, fontsize=12)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_title(rf'Fluct. $\delta {comp} / B_0$ | Step {step} | $t \approx {time_omega_ci:.2f} \Omega_{{ci}}^{{-1}}$', fontsize=14)
            ax.grid(True, linestyle=':', alpha=0.5)
            
            outname = self.outdir / f'{comp}fluct_step{step}_{plane}.png'
            fig.savefig(outname, dpi=200, bbox_inches='tight')
            plt.close(fig)

        return True

    def batch_plot(self, steps=None, plane='xy', slice_index=None):
        if not steps:
            steps = sorted(self.file_map.keys())
            
        if not steps:
            print("No se encontraron pasos para procesar.")
            return []
            
        if self.fixed_scale:
            self.compute_global_scales(steps, plane, slice_index)
            
        for step in steps:
            print(f"Procesando step {step}...")
            self.plot_snapshot(step, plane, slice_index)

    def create_gifs(self, plane='xy', duration=0.2):
        print("Generando GIFs...")
        for comp in (['Bmag'] if self.comp_magnitude else []) + self.comps_to_plot:
            pattern = f"Bmag_fluct_step*_{plane}.png" if comp == 'Bmag' else f"{comp}fluct_step*_{plane}.png"
            files = sorted(self.outdir.glob(pattern), key=lambda x: int(re.search(r'step(\d+)_', x.name).group(1)))
            
            if not files:
                continue
                
            gif_path = self.outdir / f"{comp}_fluct_{plane}.gif"
            with imageio.get_writer(gif_path, mode='I', duration=duration, loop=0) as writer:
                for file in files:
                    writer.append_data(imageio.imread(file))
            print(f"  GIF creado: {gif_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern', type=str, default='pfd.*.bp', help='Patrón de archivos ADIOS2 (*.bp)')
    parser.add_argument('--B0', type=float, default=0.01)
    parser.add_argument('--fluct', type=float, default=0.1)
    parser.add_argument('--plane', type=str, default='xy', choices=['xy','xz','yz'])
    parser.add_argument('--steps', nargs='*', type=int)
    parser.add_argument('--slice', type=int, default=None)
    parser.add_argument('--outdir', type=str, default='field_images')
    parser.add_argument('--smooth', type=float, default=0.8)
    parser.add_argument('--no_dyn_scale', action='store_true')
    parser.add_argument('--no_mag', action='store_true')
    parser.add_argument('--comps', nargs='+', default=['Bx','By','Bz'], choices=['Bx','By','Bz'])
    parser.add_argument('--fixed_scale', action='store_true')
    parser.add_argument('--create_gifs', action='store_true')
    parser.add_argument('--gif_duration', type=float, default=0.2)
    args = parser.parse_args()

    if args.B0 == 0:
        args.B0 = 0.01

    plotter = FieldImagePlotter(
        em_pattern=args.pattern,
        B0=args.B0,
        fluct_amp=args.fluct,
        outdir=args.outdir,
        smooth_sigma=args.smooth,
        dyn_scale=not args.no_dyn_scale,
        comp_magnitude=not args.no_mag,
        comps_to_plot=args.comps,
        fixed_scale=args.fixed_scale
    )
    
    plotter.batch_plot(steps=args.steps, plane=args.plane, slice_index=args.slice)
    
    if args.create_gifs:
        plotter.create_gifs(plane=args.plane, duration=args.gif_duration)
