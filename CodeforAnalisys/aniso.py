import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import glob
import os
import re
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class PlasmaAnalyzer:
    def __init__(self, n0=1e19, T0=1e6, B0_ref=0.01, outdir="outsimage"):
        """
        Inicializa el analizador con factores de normalización.
        
        Parámetros:
        -----------
        n0 : float
            Densidad de normalización [m^-3]
        T0 : float  
            Temperatura de normalización [K]
        B0_ref : float
            Campo magnético de referencia [T]
        outdir : str
            Directorio de salida para imágenes
        """
        # Factores de normalización
        self.n0 = n0
        self.T0 = T0
        self.B0_ref = B0_ref
        
        # Constantes físicas
        self.kB = 1.380649e-23  # J/K
        self.mu0 = 4 * np.pi * 1e-7  # H/m
        
        # Configuración de salida
        self.outdir = Path(outdir)
        self.outdir.mkdir(exist_ok=True)
        
        # Datos para análisis
        self.reset_data()
        
    def reset_data(self):
        """Reinicia las listas de datos."""
        self.all_anisotropy = []
        self.all_beta_par = []
        self.evol_anisotropy_mean = []
        self.evol_anisotropy_std = []
        self.evol_beta_par_mean = []
        self.evol_beta_par_std = []
        self.frac_firehose = []
        self.frac_mirror = []
        self.snapshots = []
        
    def get_uid_group(self, f, base):
        """Encuentra el grupo con el prefijo dado en el archivo HDF5."""
        for k in f.keys():
            if k.startswith(base):
                return k
        raise ValueError(f"No hay grupo {base} en archivo.")
    
    def get_matching_bzfile(self, idx, bz_files):
        """Encuentra el archivo Bz correspondiente al índice dado."""
        for f in bz_files:
            if idx in f:
                return f
        return None
    
    def calculate_instability_thresholds(self, beta_par, anisotropy):
        """
        Calcula las fracciones de plasma inestable según criterios de Firehose y Mirror.
        
        Parámetros:
        -----------
        beta_par : array
            Parámetro beta paralelo
        anisotropy : array
            Anisotropía de temperatura (T_perp/T_par)
            
        Retorna:
        --------
        tuple : (fracción_firehose, fracción_mirror)
        """
        # Evitar divisiones por cero y valores inválidos
        valid_beta = np.where(beta_par > 0, beta_par, np.nan)
        
        # Firehose: solo aplicable cuando beta_par > 2
        firehose_threshold = np.where(
            valid_beta > 2.0, 
            1 - 2 / valid_beta, 
            np.nan
        )
        firehose = np.logical_and(
            anisotropy < firehose_threshold,
            ~np.isnan(firehose_threshold)
        )
        
        # Mirror: aplicable para todo beta_par > 0
        mirror_threshold = 1 + 1 / valid_beta
        mirror = np.logical_and(
            anisotropy > mirror_threshold,
            ~np.isnan(mirror_threshold))
        
        return np.nanmean(firehose), np.nanmean(mirror)
    
    def process_snapshot(self, mom_file, bz_file):
        """
        Procesa un snapshot individual de la simulación.
        
        Parámetros:
        -----------
        mom_file : str
            Archivo de momentos
        bz_file : str
            Archivo de campo magnético
            
        Retorna:
        --------
        dict : Datos procesados del snapshot
        """
        try:
            with h5py.File(mom_file, "r") as f_mom, h5py.File(bz_file, "r") as f_bz:
                # Obtener grupos de datos
                group_mom = self.get_uid_group(f_mom, "all_1st")
                group_bz = self.get_uid_group(f_bz, "jeh-uid")
                
                # Leer datos de temperatura y densidad
                Txx = f_mom[f"{group_mom}/txx_i/p0/3d"][()]
                Tyy = f_mom[f"{group_mom}/tyy_i/p0/3d"][()]
                Tzz = f_mom[f"{group_mom}/tzz_i/p0/3d"][()]
                n = f_mom[f"{group_mom}/rho_i/p0/3d"][()]
                
                # Leer campo magnético
                Bz = f_bz[f"{group_bz}/hz_fc/p0/3d"][()]
                
        except Exception as e:
            print(f"Error leyendo archivos {mom_file}, {bz_file}: {e}")
            return None
            
        # Normalizar datos
        Txx *= self.T0
        Tyy *= self.T0
        Tzz *= self.T0
        n *= self.n0
        Bz *= self.B0_ref
        
        # Calcular temperaturas paralela y perpendicular
        Tpar = Tzz.ravel()
        Tperp = 0.5 * (Txx.ravel() + Tyy.ravel())
        n_flat = n.ravel()
        Bz_flat = Bz.ravel()
        
        # Calcular anisotropía y parámetro beta
        anisotropy = Tperp / (Tpar + 1e-30)  # Evitar división por cero
        Ppar = n_flat * self.kB * Tpar
        Pmag = Bz_flat**2 / (2 * self.mu0)
        beta_par = Ppar / (Pmag + 1e-30)
        
        # Filtrar datos válidos
        mask = (Tpar > 0) & (Tperp > 0) & (n_flat > 0) & \
               (np.abs(Bz_flat) > 1e-30) & np.isfinite(beta_par) & \
               np.isfinite(anisotropy)
        
        return {
            'anisotropy': anisotropy[mask],
            'beta_par': beta_par[mask],
            'Tpar': Tpar[mask],
            'Tperp': Tperp[mask],
            'n': n_flat[mask],
            'Bz': Bz_flat[mask],
            'mask_size': np.sum(mask),
            'total_size': len(mask)
        }
    
    def analyze_simulation(self, mom_pattern="pfd_moments.*.h5", bz_pattern="pfd.*.h5"):
        """
        Analiza toda la simulación procesando todos los snapshots.
        
        Parámetros:
        -----------
        mom_pattern : str
            Patrón para archivos de momentos
        bz_pattern : str
            Patrón para archivos de campo magnético
        """
        print("Iniciando análisis de simulación...")
        
        # Buscar archivos
        mom_files = sorted(glob.glob(mom_pattern))
        bz_files = sorted(glob.glob(bz_pattern))
        
        if not mom_files:
            raise FileNotFoundError(f"No se encontraron archivos con patrón {mom_pattern}")
        if not bz_files:
            raise FileNotFoundError(f"No se encontraron archivos con patrón {bz_pattern}")
            
        print(f"Encontrados {len(mom_files)} archivos de momentos y {len(bz_files)} de campo B")
        
        # Reiniciar datos
        self.reset_data()
        
        processed_count = 0
        for i, mf in enumerate(mom_files):
            # Extraer índice del archivo
            idx_match = re.search(r'\.(\d+)_', mf)
            idx = idx_match.group(1) if idx_match else str(i)
            
            # Buscar archivo Bz correspondiente
            bzfile = self.get_matching_bzfile(idx, bz_files)
            if not bzfile:
                print(f"No se encontró archivo Bz para snapshot {idx}")
                continue
                
            print(f"Procesando snapshot {idx} ({i+1}/{len(mom_files)})...")
            
            # Procesar snapshot
            data = self.process_snapshot(mf, bzfile)
            if data is None:
                continue
                
            # Acumular datos globales
            self.all_anisotropy.extend(data['anisotropy'])
            self.all_beta_par.extend(data['beta_par'])
            
            # Calcular estadísticas del snapshot
            if len(data['anisotropy']) > 0:
                self.evol_anisotropy_mean.append(np.mean(data['anisotropy']))
                self.evol_anisotropy_std.append(np.std(data['anisotropy']))
                self.evol_beta_par_mean.append(np.mean(data['beta_par']))
                self.evol_beta_par_std.append(np.std(data['beta_par']))
                self.snapshots.append(int(idx))
                
                # Calcular fracciones de inestabilidad
                frac_fire, frac_mirr = self.calculate_instability_thresholds(
                    data['beta_par'], data['anisotropy'])
                self.frac_firehose.append(frac_fire)
                self.frac_mirror.append(frac_mirr)
                
                processed_count += 1
                
        print(f"Procesados exitosamente {processed_count} snapshots")
        
        # Convertir a arrays numpy para facilitar cálculos
        self.all_anisotropy = np.array(self.all_anisotropy)
        self.all_beta_par = np.array(self.all_beta_par)
        
    def print_statistics(self):
        """Imprime estadísticas generales de la simulación."""
        if len(self.all_anisotropy) == 0:
            print("No hay datos para mostrar estadísticas")
            return
            
        print("\n" + "="*50)
        print("ESTADÍSTICAS GENERALES")
        print("="*50)
        print(f"Total de puntos analizados: {len(self.all_anisotropy):,}")
        print(f"Número de snapshots: {len(self.snapshots)}")
        print()
        print(f"Anisotropía (T⊥/T∥):")
        print(f"  Min/Max/Media: {np.min(self.all_anisotropy):.3f} / {np.max(self.all_anisotropy):.3f} / {np.mean(self.all_anisotropy):.3f}")
        print(f"  Desviación estándar: {np.std(self.all_anisotropy):.3f}")
        print()
        print(f"Beta paralelo:")
        print(f"  Min/Max/Media: {np.min(self.all_beta_par):.3e} / {np.max(self.all_beta_par):.3e} / {np.mean(self.all_beta_par):.3e}")
        print(f"  Desviación estándar: {np.std(self.all_beta_par):.3e}")
        print()
        if self.frac_firehose:
            print(f"Fracción promedio Firehose: {np.mean(self.frac_firehose):.4f}")
            print(f"Fracción promedio Mirror: {np.mean(self.frac_mirror):.4f}")
    
    def plot_scatter_anisotropy_beta_unified(self, save=True, show=False):
        """Genera gráfico de dispersión anisotropía vs beta paralelo con evolución temporal (colores invertidos y línea de tendencia)"""
        plt.figure(figsize=(12, 8))
        
        # Crear arrays para almacenar datos con información temporal
        all_beta_temporal = []
        all_aniso_temporal = []
        all_times = []
        
        # Arrays para calcular la línea de tendencia temporal
        time_means_beta = []
        time_means_aniso = []
        time_points = []
        
        # Recopilar datos con información temporal
        mom_files = sorted(glob.glob("pfd_moments.*.h5"))
        for i, mf in enumerate(mom_files):
            idx_match = re.search(r'\.(\d+)_', mf)
            idx = idx_match.group(1) if idx_match else None
            bz_files = sorted(glob.glob("pfd.*.h5"))
            bzfile = self.get_matching_bzfile(idx, bz_files) if idx else None
            if not bzfile:
                continue
                
            data = self.process_snapshot(mf, bzfile)
            if data is None or len(data['anisotropy']) == 0:
                continue
                
            # Agregar datos con marca temporal
            all_beta_temporal.extend(data['beta_par'])
            all_aniso_temporal.extend(data['anisotropy'])
            # Crear array de tiempo normalizado (0 = inicio, 1 = final)
            time_normalized = i / max(1, len(mom_files) - 1)
            all_times.extend([time_normalized] * len(data['anisotropy']))
            
            # Calcular medias para línea de tendencia
            time_means_beta.append(np.mean(data['beta_par']))
            time_means_aniso.append(np.mean(data['anisotropy']))
            time_points.append(time_normalized)
        
        all_beta_temporal = np.array(all_beta_temporal)
        all_aniso_temporal = np.array(all_aniso_temporal)
        all_times = np.array(all_times)
        
        # Submuestrear si hay muchos puntos
        n_points = len(all_beta_temporal)
        if n_points > 30000:
            indices = np.random.choice(n_points, 30000, replace=False)
            beta_plot = all_beta_temporal[indices]
            aniso_plot = all_aniso_temporal[indices]
            times_plot = all_times[indices]
            alpha = 0.6
            size = 2
        else:
            beta_plot = all_beta_temporal
            aniso_plot = all_aniso_temporal
            times_plot = all_times
            alpha = 0.7
            size = 3
        
        # Crear colormap personalizado: morado oscuro (#4A0E4E) a verde fuerte (#27AE60) - INVERTIDO
        colors = ["#4A0E4E", "#27AE60"]  # Morado más oscuro a verde fuerte (invertido)
        cmap_custom = LinearSegmentedColormap.from_list("purple_green", colors)
        
        # Crear scatter plot con colormap personalizado
        scatter = plt.scatter(
            beta_plot, 
            aniso_plot, 
            c=times_plot, 
            s=size, 
            alpha=alpha, 
            cmap=cmap_custom,
            rasterized=True
        )
        
        # Línea de tendencia temporal (camino promedio de las partículas)
        if len(time_means_beta) > 1:
            time_means_beta = np.array(time_means_beta)
            time_means_aniso = np.array(time_means_aniso)
            time_points = np.array(time_points)
            
            # Ordenar por tiempo para la línea
            sort_idx = np.argsort(time_points)
            beta_trend = time_means_beta[sort_idx]
            aniso_trend = time_means_aniso[sort_idx]
            time_sorted = time_points[sort_idx]
            
            # Plotear línea de tendencia como una sola línea blanca
            plt.plot(beta_trend, aniso_trend, '-', color='white', linewidth=4, 
                    alpha=0.9, label='Trayectoria promedio', zorder=5)
            
            # Marcar inicio y final de la trayectoria
            plt.scatter(beta_trend[0], aniso_trend[0], s=100, color='#4A0E4E', 
                       marker='o', edgecolor='white', linewidth=2, 
                       label='Inicio (t=0)', zorder=10)
            plt.scatter(beta_trend[-1], aniso_trend[-1], s=100, color='#27AE60', 
                       marker='s', edgecolor='white', linewidth=2, 
                       label='Final (t=T)', zorder=10)
        
        # Barra de colores con etiquetas invertidas
        cbar = plt.colorbar(scatter, label='Progreso Temporal')
        cbar.set_label('Tiempo (temprano→tardío)', fontsize=12)
        
        plt.xlabel(r'$\beta_\parallel$', fontsize=14)
        plt.ylabel(r'$T_\perp / T_\parallel$', fontsize=14)
        plt.title('Anisotropía vs Beta Paralelo)', fontsize=16)
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Líneas de referencia
        plt.axhline(1, color='red', linestyle='--', alpha=0.8, label='Isotropía')
        
        # Criterios de inestabilidad mejorados
        # Firehose: solo para beta_par > 2
        beta_firehose = np.logspace(np.log10(2.1), 3, 100)
        plt.plot(
            beta_firehose, 
            1 - 2/beta_firehose, 
            color='#ff7f00', 
            linestyle='-', 
            linewidth=2.5,
            label=r'Firehose: $T_\perp/T_\parallel < 1 - 2/\beta_\parallel$'
        )
        
        # Mirror: para todo beta_par > 0
        beta_mirror = np.logspace(-1, 3, 100)
        plt.plot(
            beta_mirror, 
            1 + 1/beta_mirror, 
            color='#377eb8', 
            linestyle='-', 
            linewidth=2.5,
            label=r'Mirror: $T_\perp/T_\parallel > 1 + 1/\beta_\parallel$'
        )
        
        plt.legend(fontsize=11, loc='upper right')
        plt.tight_layout()
        
        if save:
            filename = self.outdir / "anisotropia_vs_beta_unified.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado: {filename}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_temporal_evolution(self, save=True, show=False):
        """Genera gráficos de evolución temporal."""
        if not self.snapshots:
            print("No hay datos para mostrar evolución temporal")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Evolución de anisotropía
        ax1.errorbar(self.snapshots, self.evol_anisotropy_mean, 
                    yerr=self.evol_anisotropy_std, fmt='o-', capsize=3)
        ax1.set_xlabel('Snapshot')
        ax1.set_ylabel(r'$\langle T_\perp / T_\parallel \rangle$')
        ax1.set_title('Evolución de Anisotropía Promedio')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(1, color='gray', linestyle='--', alpha=0.7)
        
        # Evolución de beta paralelo
        ax2.errorbar(self.snapshots, self.evol_beta_par_mean, 
                    yerr=self.evol_beta_par_std, fmt='s-', capsize=3)
        ax2.set_xlabel('Snapshot')
        ax2.set_ylabel(r'$\langle \beta_\parallel \rangle$')
        ax2.set_title('Evolución de Beta Paralelo Promedio')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        # Fracciones de inestabilidad
        ax3.plot(self.snapshots, self.frac_firehose, 'r-o', label='Firehose', markersize=4)
        ax3.plot(self.snapshots, self.frac_mirror, 'b-s', label='Mirror', markersize=4)
        ax3.set_xlabel('Snapshot')
        ax3.set_ylabel('Fracción Inestable')
        ax3.set_title('Evolución de Inestabilidades')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Histograma de anisotropía
        ax4.hist(self.all_anisotropy, bins=50, alpha=0.7, density=True)
        ax4.axvline(1, color='gray', linestyle='--', alpha=0.7, label='Isotropía')
        ax4.axvline(np.mean(self.all_anisotropy), color='red', linestyle='-', 
                   alpha=0.8, label=f'Media = {np.mean(self.all_anisotropy):.2f}')
        ax4.set_xlabel(r'$T_\perp/T_\parallel$')
        ax4.set_ylabel('Densidad de Probabilidad')
        ax4.set_title('Distribución de Anisotropía')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.outdir / "evolucion_temporal.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado: {filename}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_histograms(self, save=True, show=False):
        """Genera histogramas de las distribuciones."""
        if len(self.all_anisotropy) == 0:
            print("No hay datos para mostrar histogramas")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Histograma de anisotropía
        ax1.hist(self.all_anisotropy, bins=50, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax1.axvline(1, color='red', linestyle='--', linewidth=2, label='Isotropía')
        ax1.axvline(np.mean(self.all_anisotropy), color='orange', linestyle='-', 
                   linewidth=2, label=f'Media = {np.mean(self.all_anisotropy):.2f}')
        ax1.set_xlabel(r'$T_\perp/T_\parallel$')
        ax1.set_ylabel('Frecuencia')
        ax1.set_title('Distribución de Anisotropía')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Histograma de beta paralelo (escala log)
        ax2.hist(self.all_beta_par, bins=np.logspace(np.log10(max(np.min(self.all_beta_par), 1e-6)), 
                                                    np.log10(np.max(self.all_beta_par)), 50), 
                alpha=0.7, edgecolor='black', linewidth=0.5)
        ax2.set_xlabel(r'$\beta_\parallel$')
        ax2.set_ylabel('Frecuencia')
        ax2.set_title('Distribución de Beta Paralelo')
        ax2.set_xscale('log')
        ax2.set_yscale('log')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filename = self.outdir / "histogramas.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Gráfico guardado: {filename}")
            
        if show:
            plt.show()
        else:
            plt.close()
    
    def run_full_analysis(self, mom_pattern="pfd_moments.*.h5", bz_pattern="pfd.*.h5", 
                         save_plots=True, show_plots=False):
        """
        Ejecuta el análisis completo de la simulación.
        
        Parámetros:
        -----------
        mom_pattern : str
            Patrón para archivos de momentos
        bz_pattern : str
            Patrón para archivos de campo magnético
        save_plots : bool
            Si guardar los gráficos
        show_plots : bool
            Si mostrar los gráficos
        """
        print("INICIANDO ANÁLISIS COMPLETO DE SIMULACIÓN DE PLASMA")
        print("="*60)
        
        # Analizar simulación
        self.analyze_simulation(mom_pattern, bz_pattern)
        
        # Mostrar estadísticas
        self.print_statistics()
        
        # Generar gráficos
        print("\nGenerando gráficos...")
        self.plot_scatter_anisotropy_beta_unified(save=save_plots, show=show_plots)
        self.plot_temporal_evolution(save=save_plots, show=show_plots)
        self.plot_histograms(save=save_plots, show=show_plots)
        
        print(f"\nAnálisis completado. Resultados guardados en: {self.outdir}")
        
        return {
            'anisotropy_stats': {
                'mean': np.mean(self.all_anisotropy),
                'std': np.std(self.all_anisotropy),
                'min': np.min(self.all_anisotropy),
                'max': np.max(self.all_anisotropy)
            },
            'beta_stats': {
                'mean': np.mean(self.all_beta_par),
                'std': np.std(self.all_beta_par),
                'min': np.min(self.all_beta_par),
                'max': np.max(self.all_beta_par)
            },
            'instability_fractions': {
                'firehose': np.mean(self.frac_firehose) if self.frac_firehose else 0,
                'mirror': np.mean(self.frac_mirror) if self.frac_mirror else 0
            }
        }


# Ejemplo de uso
if __name__ == "__main__":
    # Crear analizador con parámetros por defecto
    analyzer = PlasmaAnalyzer(
        n0=1e19,      # Densidad de normalización [m^-3]
        T0=1e6,       # Temperatura de normalización [K]  
        B0_ref=0.01,  # Campo magnético de referencia [T]
        outdir="outsimage"
    )
    
    # Ejecutar análisis completo
    results = analyzer.run_full_analysis(
        mom_pattern="pfd_moments.*.h5",
        bz_pattern="pfd.*.h5",
        save_plots=True,
        show_plots=False  # Cambiar a True para mostrar gráficos
    )
    
    print("\nResultados finales:")
    print(f"Anisotropía promedio: {results['anisotropy_stats']['mean']:.3f}")
    print(f"Beta paralelo promedio: {results['beta_stats']['mean']:.3e}")
    print(f"Fracción Firehose: {results['instability_fractions']['firehose']:.4f}")
    print(f"Fracción Mirror: {results['instability_fractions']['mirror']:.4f}")
