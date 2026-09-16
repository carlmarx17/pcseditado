"""Regression tests for real box geometry, window density and streamed spectra."""
import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import h5py
import numpy as np
from run_geometry import resolve_run
from streaming_fields import SnapshotSeries, spatial_spectra
from validate_moments import measure_particles, validation_rows, M_ION, M_ELEC
from polarization_dispersion import stream_polarization, build_psi, spatial_fft_kpar0
from growth_rate_map import compute_growth_rate_map
from dispersion_analysis import compute_phase_velocity_density


class RunAwareTests(unittest.TestCase):
    def profile(self):
        return dict(ngrid=576,domain_di=20.,mass_ratio=200.,nicell=1000,nmax=1200000)

    def grid_file(self, directory, n, length):
        p=Path(directory)/'pfd.000000_p000000.h5'
        with h5py.File(p,'w') as f:
            f.create_dataset('jeh-0/hx_fc/p0/3d',shape=(n,n,1),dtype='f4')
            for a in (1,2):
                f.create_dataset(f'crd[{a}]/p0/1d',data=(np.arange(n)+.5)*length*np.sqrt(200)/n)
        return p

    def test_actual_grid_and_domain_576_and_1152(self):
        for n,length in [(576,20),(1152,40),(288,20)]:
            with tempfile.TemporaryDirectory() as d:
                self.grid_file(d,n,length)
                with patch.dict(os.environ,{'PSC_ANALYSIS_DATA_DIR':d,'PSC_ANALYSIS_CONFIG':''}):
                    values,origin=resolve_run(self.profile())
                self.assertEqual(values['ngrid'],n)
                self.assertAlmostEqual(values['domain_di'],length)
                self.assertNotEqual(origin['domain_di'],'profile')

    def test_conflicting_geometry_rejected_and_runtime_dt_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            self.grid_file(d,1152,40)
            p=Path(d)/'runtime.json';p.write_text(json.dumps(dict(ngrid=576)))
            with patch.dict(os.environ,{'PSC_ANALYSIS_DATA_DIR':d,'PSC_ANALYSIS_CONFIG':str(p)}):
                with self.assertRaisesRegex(ValueError,'conflicts'):
                    resolve_run(self.profile())
                p.write_text(json.dumps(dict(ngrid=1152,dt_code=.125,nicell=1500)))
                values,_=resolve_run(self.profile())
                self.assertEqual(values['dt_code'],.125)
                self.assertEqual(values['nicell'],1500)

    def test_density_weights_and_window_independent_of_chunks(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'prt_case.000000.h5'
            dtype=[(k,'f8') for k in ['q','m','px','py','pz','w']]
            a=np.zeros(8,dtype=dtype)
            a['q']=[1]*4+[-1]*4;a['m']=[M_ION]*4+[M_ELEC]*4
            a['w']=[500,1500,500,1500]*2
            a['px']=a['py']=a['pz']=[-.1,.1,.1,-.1]*2
            with h5py.File(p,'w') as f:
                g=f.create_group('particles');g.attrs['lo']=[0,4,7];g.attrs['hi']=[1,6,9]
                f.create_dataset('particles/p0/1d',data=a)
            for chunk in [1,3,100]:
                m,w=measure_particles(p,cori=.001,chunk_size=chunk)
                self.assertEqual(w['sampled_cells'],4)
                self.assertAlmostEqual(m['ion']['n'],1.)
                self.assertAlmostEqual(m['electron']['n'],1.)
                self.assertTrue(all(r['status']=='NOT_INITIAL' for r in validation_rows(m,570000)))
                self.assertTrue(all(r['status']=='PASS' for r in validation_rows(m,0) if r['quantity']=='n'))

    def wave(self):
        times=np.linspace(0,8,33);z,y=np.meshgrid(np.arange(32),np.arange(32),indexing='ij')
        fields=np.zeros((3,len(times),32,32))
        phase=2*np.pi*(2*z+y)/32
        for i,t in enumerate(times):
            fields[:,i]=np.array([np.cos(phase-.6*t),np.sin(phase-.6*t),np.cos(phase)])*np.exp(.12*t)
            fields[2,i]+=10
        return fields,times

    def test_streamed_fft_equals_full_transform(self):
        fields,times=self.wave();seen=[]
        def loader(i):seen.append(i);return fields[:,i]
        lazy=SnapshotSeries(range(len(times)),loader)
        sl=slice(13,20)
        for window in [np.ones((32,32)),np.outer(np.hanning(32),np.hanning(32))]:
            result=spatial_spectra(lazy,window,sl,sl)
            full=np.fft.fftshift(np.fft.fft2((fields-fields.mean(axis=(2,3),keepdims=True))*window,axes=(2,3)),axes=(2,3))
            np.testing.assert_allclose(result,full[:,:,sl,sl],atol=1e-11)
        self.assertEqual(max(seen),len(times)-1)

    def test_polarized_stream_equals_normalized_full_fft(self):
        fields,times=self.wave()
        # Add a k_perp=0 signal, as the oblique wave must average away.
        z=np.arange(32)[:,None]
        fields[0]+=np.cos(2*np.pi*z/32)[None]
        lazy=SnapshotSeries(range(len(times)),lambda i:fields[:,i])
        ap,am,k,result=stream_polarization(lazy,(.4,.4),('z','y'),'z',.08,mirror=True)
        plus,minus=build_psi(fields[0],fields[1],.08)
        for measured,expected in [(ap,plus),(am,minus)]:
            direct,k2=spatial_fft_kpar0(expected,(.4,.4),('z','y'),'z')
            np.testing.assert_allclose(measured,direct,atol=1e-12)
            np.testing.assert_allclose(k,k2)
        self.assertGreaterEqual(result['theta_kb_deg'],45)

    def test_streamed_growth_recovers_injected_rate(self):
        fields,times=self.wave();lazy=SnapshotSeries(range(len(times)),lambda i:fields[:,i])
        result=compute_growth_rate_map(lazy,times,(1,1),('z','y'),kpar_max=1,kperp_max=1)
        idx=np.unravel_index(np.argmax(result['final_power']),result['final_power'].shape)
        self.assertAlmostEqual(result['gamma'][idx],.12,places=10)


if __name__=='__main__':unittest.main()
