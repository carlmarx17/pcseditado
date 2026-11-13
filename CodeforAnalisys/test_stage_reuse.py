import json

from analysis_contract import inventory_identity
from run_pipeline import is_complete
from stage_reuse import import_stages
from submit_missing import HERE


def test_reuse_requires_matching_history_intact_products_and_options(tmp_path):
    data, src, out = [tmp_path/p for p in ('raw', 'archive', 'out')]
    for p in (data, src, out):
        p.mkdir()
    manifest = src/'case_analysis_manifest.json'
    manifest.write_text(json.dumps({'input_directory':str(data)}))
    f = src/'result.csv'
    f.write_text('x\n1\n')
    entry = {'execution_status':'PASS','products':[f.name],'output_identity':inventory_identity([f])}
    sig = {'input':'same','source':'same','case':'case','config':None,'environment':{},'options':[]}
    (src/'pipeline.json').write_text(json.dumps({'signature':sig, 'stages':{'structures':entry}}))
    state = {'signature':dict(sig, options=['STRUCTURES_FLAGS=--attribution']), 'stages':{}}
    def attempt():
        import_stages([src], data, out, state, ['structures'], HERE, is_complete)
    attempt()
    assert not state['stages']
    state['signature'] = sig
    attempt()
    assert (out/f.name).resolve() == f
    assert is_complete(state['stages']['structures'], out)
    f.write_text('x\n100\n')
    assert not is_complete(state['stages']['structures'], out)



def test_partial_production_histories_and_family_selection(tmp_path, monkeypatch):
    import submit_missing as planner
    raw, archives, out = [tmp_path/p for p in ('raw', 'archives', 'out')]
    raw.mkdir()
    archives.mkdir()
    cases = ('firehose_bikappa3_bigbox40', 'mirror_bikappa3_moderate', 'whistler_bimaxwellian_strong_mr800')
    for case in cases:
        data = raw/('psc_'+case+'_123')
        data.mkdir()
        (data/'pfd.000001.bp').mkdir()
    monkeypatch.setattr(planner.subprocess, 'check_output', lambda *a, **kw: '919000 1200000 1200000 500')
    monkeypatch.setattr(planner, 'provenance', lambda: {'analysis_source_sha256':'test'})
    rows = planner.inventory(raw, archives, out)
    assert len(rows) == 3
    assert all(r['coverage']['partial'] for r in rows)
    assert all('heating-budget' in r['stages'] for r in rows)
    rows = planner.inventory(raw, archives, out, 'firehose')
    assert [r['case'] for r in rows] == [cases[0]]
    assert rows[0]['coverage']['last_step'] == 919000
