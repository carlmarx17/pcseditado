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
