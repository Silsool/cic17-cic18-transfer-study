"""Portable notebook and synthetic end-to-end checks; no private data or network."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

REPO=Path(__file__).resolve().parents[1]

class NotebookChecks(unittest.TestCase):
    def test_clean_notebook_sources(self):
        files=sorted((REPO/'notebooks').glob('*.ipynb'))
        self.assertEqual(len(files),5)
        for path in files:
            nb=json.loads(path.read_text())
            ids=[c['id'] for c in nb['cells']]
            self.assertEqual(len(ids),len(set(ids)))
            self.assertEqual(nb['cells'][0]['id'],'study-paths')
            for cell in nb['cells']:
                source=''.join(cell['source'])
                self.assertNotIn('/home/',source)
                if cell['cell_type']=='code':
                    compile(source,path.name+':'+cell['id'],'exec')
                    self.assertEqual(cell['outputs'],[])
                    self.assertIsNone(cell['execution_count'])

    def test_startup_paths(self):
        nb=json.loads((REPO/'notebooks/binary_classification_baseline.ipynb').read_text())
        source=''.join(nb['cells'][0]['source'])
        original=Path.cwd()
        with tempfile.TemporaryDirectory() as tmp:
            env={'NIDS_DATA_DIR':tmp,'NIDS_ARTIFACT_DIR':str(Path(tmp)/'outputs')}
            try:
                for cwd in [REPO,REPO/'notebooks',REPO.parent]:
                    os.chdir(cwd)
                    with patch.dict(os.environ,env):
                        namespace={}
                        exec(source,namespace)
                        self.assertEqual(namespace['REPO_ROOT'],REPO)
                        self.assertEqual(namespace['DATA_DIR'],Path(tmp))
                        self.assertEqual(namespace['ARTIFACT_DIR'],Path(tmp)/'outputs')
            finally:
                os.chdir(original)

class PipelineChecks(unittest.TestCase):
    def test_binary_transfer(self):
        import sys,tempfile
        from pathlib import Path
        import numpy as np,pandas as pd,duckdb,joblib
        from nids_transfer.binary_transfer_utils import BinaryStudy,fit_coverage,coverage_reference,select_coverage_queries,assess_coverage,coverage_reports
        rng=np.random.default_rng(6)
        with tempfile.TemporaryDirectory(dir='/tmp') as root:
         root=Path(root);paths={};original={}
         for year,n in [('2017',1200),('2018',600)]:
          attack=rng.integers(0,2,n)
          df=pd.DataFrame({'id':range(n),'Src IP':['ip']*n,'Flow Duration':rng.exponential(3+attack*10),
              'Packet Length Mean':rng.exponential(5+attack*30),'Fwd Header Length':rng.exponential(4,n),
              'Total Fwd Packet':rng.integers(1,100,n),'SYN Flag Count':rng.integers(0,5,n),
              'Protocol':rng.choice([6,17] if year=='2017' else [6,17,99],n),
              'Label':np.where(attack,'Attack','BENIGN')})
          df.loc[0,'Fwd Header Length']=-1
          if year=='2018':df.loc[10,'Flow Duration']=np.inf
          c=duckdb.connect();c.register('data',df);path=root/f'{year}.parquet';c.execute(f"COPY data TO '{path}' (FORMAT PARQUET)");c.close()
          paths[year]=path;original[year]=df
         study=BinaryStudy(paths,root/'artifacts',batch_size=127)
         counts=study.freeze_splits();assert counts['train_2017']==960 and counts['test_2018']==599
         stats=study.prepare_matrices();p=joblib.load(study.out/'preprocessor.joblib')
         train=pd.read_parquet(study.out/'train_2017_rows.parquet')
         assert not set(train.source_row)&set(pd.read_parquet(study.out/'test_2017_rows.parquet').source_row)
         raw=original['2017'].iloc[train.source_row][study.numeric].to_numpy(dtype=float,copy=True);raw[raw<0]=np.nan
         assert np.allclose(np.nanmean(np.log1p(raw),axis=0),p['mean'])
         expected,_=study.transform(original['2017'].iloc[train.source_row],p)
         assert np.array_equal(expected,np.load(study.out/'train_2017_X.npy'))
         assert stats['test_2018']['unknown_protocol']>0
         for name in ['logreg','rf']:
          study.fit_model(name);overall,labels=study.predict(name);assert len(overall)==4 and len(labels)==4
          pred=pd.read_parquet(study.out/f'{name}_test_2018_predictions.parquet');assert len(pred)==599
         space=fit_coverage(study);ref=coverage_reference(study,space,calibration_rows=100,k_neighbors=3)
         queries=select_coverage_queries(study,population_per_split=100,minimum_per_label=60)
         cov=assess_coverage(study,space,ref,queries)
         assert len(cov)==len(queries) and cov.coverage_distance.ge(0).all()
         assert cov.groupby('split').population_sample.sum().eq(100).all()
         summary,perlabel=coverage_reports(study,cov)
         assert summary.query("scope == 'all'").groupby(['model','split']).rows.sum().eq(100).all()
         assert ref['training_rows']==960 and ref['q99']>ref['q90']
         for name in ['logreg','rf']:
          joined=pd.read_parquet(study.out/'coverage'/f'{name}_coverage_predictions.parquet')
          assert joined.row_key.is_unique
         print('PASS: train-only normalization; frozen disjoint splits; exact matrix/row alignment; unknown protocols; full prediction counts; both models; full training coverage; self-exclusion; population-only denominators; stable prediction joins.')

    def test_clustering(self):
        import sys
        from pathlib import Path
        import tempfile
        import numpy as np
        import pandas as pd
        import duckdb
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import PCA
        from nids_transfer.dataset_clustering_utils import FlowStudy, fit_auxiliary, representation, run_experiment, evaluate, stability
        rng=np.random.default_rng(123)
        with tempfile.TemporaryDirectory(dir='/tmp') as tmp:
         paths={}
         arrays=[]
         for year in ['2017','2018']:
          n=90
          frame=pd.DataFrame({'id':range(n),'Flow ID':['f']*n,'Src IP':['a']*n,
            'Protocol':np.tile([6,17,1],30),'Label':np.tile(['BENIGN','DoS Hulk','DoS Hulk - Attempted'],30),
            'Flow Duration':rng.exponential(10,n),'Packet Length Mean':rng.exponential(5,n),
            'Fwd Header Length':rng.exponential(2,n),'SYN Flag Count':rng.integers(0,4,n)})
          frame.loc[0,'Fwd Header Length']=-1
          path=Path(tmp)/f'{year}.parquet';c=duckdb.connect();c.register('f',frame);c.execute(f"COPY f TO '{path}' (FORMAT PARQUET)");c.close()
          paths[year]=path;arrays.append(frame)
         study=FlowStudy(paths,batch_size=13)
         space=study.fit_space()
         full=pd.concat(arrays,ignore_index=True)
         x=full[study.features].to_numpy(copy=True);x[x<0]=np.nan
         logged=np.log1p(x)
         ref=StandardScaler().fit(logged)
         assert np.allclose(space['scaler'].mean_,ref.mean_)
         assert np.allclose(space['scaler'].var_,ref.var_)
         z=ref.transform(logged);z=np.nan_to_num(z,nan=0)
         assert np.allclose(space['eigenvalues'],PCA().fit(z).explained_variance_)
         a=study.sample(50,7);b=study.sample(50,7)
         assert set(a.row_key)==set(b.row_key)
         aux=fit_auxiliary(a,study)
         pair=a.iloc[[0,0]].copy();pair.iloc[0,pair.columns.get_loc('Protocol')]=6;pair.iloc[1,pair.columns.get_loc('Protocol')]=17
         embedding,_=representation(pair,study,space,aux,n_pcs=2,protocol_weight=.5)
         assert np.isclose(np.sum((embedding[0]-embedding[1])**2),.5)
         r=run_experiment('test',a,study,space,aux,n_pcs=2,min_cluster_size=5,min_samples=3)
         assert len(r['assignments'])==100
         assert stability(r,r)['ARI_assigned_in_both']==1
         assignments=r['assignments'].copy();assignments['cluster']=-1
         report=evaluate(assignments)
         assert report['overview'].iloc[0].noise_rate==1
         assert report['overview'].iloc[0].clusters==0
         print('PASS: batch scaler/PCA match sklearn; reproducible sampling and unique keys; weighted protocol distance; end-to-end HDBSCAN; stability; all-noise evaluation.')

if __name__=='__main__':
    unittest.main()
