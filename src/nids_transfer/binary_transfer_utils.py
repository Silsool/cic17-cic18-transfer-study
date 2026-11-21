"""Train-only binary baselines and PCA coverage; source Parquet files are read-only."""
from pathlib import Path
import json, hashlib, warnings, time
import numpy as np
import pandas as pd
import duckdb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score
from sklearn.neighbors import KDTree
from .dataset_clustering_utils import qi, literal, normalize, METADATA, CATEGORICAL, NUMERIC_TYPES, Moments

VERSION = 1


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, default=str) + '\n')


def read_json(path):
    return json.loads(Path(path).read_text())


class BinaryStudy:
    def __init__(self, paths, out, seed=42, batch_size=16384):
        self.paths = {str(y): Path(p).resolve() for y, p in paths.items()}
        self.out = Path(out); self.out.mkdir(parents=True, exist_ok=True)
        self.seed, self.batch_size = seed, batch_size
        self.con = duckdb.connect(config={'memory_limit': '512MB', 'threads': 2})
        schemas = {y: self.con.read_parquet(str(p)) for y, p in self.paths.items()}
        first = schemas['2017']
        assert first.columns == schemas['2018'].columns
        types = {y: dict(zip(r.columns, r.types)) for y,r in schemas.items()}
        self.numeric = [c for c in first.columns if normalize(c) not in METADATA | CATEGORICAL
                        and all(t[c].id in NUMERIC_TYPES for t in types.values())]
        self.continuous = [c for c in self.numeric if 'flag' not in normalize(c)]
        self.protocol = next(c for c in first.columns if normalize(c) == 'protocol')
        self.label = next(c for c in first.columns if normalize(c) == 'label')
        self.valid = ' AND '.join(f'coalesce(isfinite({qi(c)}), false)' for c in self.numeric)
        self.signature = {'version': VERSION, 'seed': seed, 'test_size': .2,
                          'numeric': self.numeric, 'protocol': self.protocol,
                          'inputs': {y: {'path': str(p), 'bytes': p.stat().st_size,
                                          'mtime_ns': p.stat().st_mtime_ns} for y,p in self.paths.items()}}
        manifest = self.out / 'manifest.json'
        if manifest.exists() and read_json(manifest) != self.signature:
            raise ValueError('Artifact configuration/input mismatch: choose a new output directory.')
        write_json(manifest, self.signature)

    def source(self, year):
        return f'read_parquet({literal(self.paths[year])}, file_row_number=true)'

    def batches(self, year):
        selection = ', '.join(map(qi, ['file_row_number'] + self.numeric + [self.protocol, self.label]))
        rel = self.con.sql(f'SELECT {selection} FROM {self.source(year)} WHERE {self.valid}')
        reader = rel.to_arrow_reader(batch_size=self.batch_size) if hasattr(rel, 'to_arrow_reader') else rel.fetch_record_batch(self.batch_size)
        for batch in reader: yield batch.to_pandas()

    def freeze_splits(self):
        marker = self.out / 'splits_complete.json'
        if marker.exists(): return read_json(marker)
        counts = {}
        for year in self.paths:
            meta = self.con.sql(f'SELECT file_row_number AS source_row, {qi(self.label)} AS label '
                                f'FROM {self.source(year)} WHERE {self.valid} ORDER BY source_row').df()
            if meta.label.isna().any(): raise ValueError('Missing labels need a policy before splitting')
            meta['year'] = year
            meta['row_key'] = year + ':' + meta.source_row.astype(str)
            meta['y'] = meta.label.ne('BENIGN').astype('int8')
            if year == '2017':
                vc = meta.label.value_counts()
                if vc.min() < 2: raise ValueError('Rare raw label prevents stratified row split; review grouping')
                train, test = train_test_split(np.arange(len(meta)), test_size=.2, random_state=self.seed,
                                               stratify=meta.label)
                indices = {'train_2017': train, 'test_2017': test}
            else: indices = {'test_2018': np.arange(len(meta))}
            for split, idx in indices.items():
                selected = meta.iloc[idx].sort_values('source_row').reset_index(drop=True)
                selected.to_parquet(self.out / f'{split}_rows.parquet', index=False)
                counts[split] = len(selected)
            total = self.con.sql(f'SELECT count(*) FROM {self.source(year)}').fetchall()[0][0]
            counts[f'excluded_nonfinite_{year}'] = total - len(meta)
        a = pd.read_parquet(self.out / 'train_2017_rows.parquet', columns=['source_row'])
        b = pd.read_parquet(self.out / 'test_2017_rows.parquet', columns=['source_row'])
        assert np.intersect1d(a.source_row, b.source_row).size == 0
        write_json(marker, counts)
        return counts

    def train_lookup(self):
        rows = pd.read_parquet(self.out / 'train_2017_rows.parquet', columns=['source_row']).source_row.to_numpy()
        size = self.con.sql(f'SELECT count(*) FROM {self.source("2017")}').fetchall()[0][0]
        lookup = np.zeros(size, dtype=bool);lookup[rows] = True
        return lookup

    def fit_preprocessor(self):
        artifact = self.out / 'preprocessor.joblib'
        if artifact.exists(): return joblib.load(artifact)
        is_train = self.train_lookup()
        scalers = [StandardScaler() for _ in self.numeric]
        protocols, neg, n = set(), np.zeros(len(self.numeric), dtype=np.int64), 0
        for batch in self.batches('2017'):
            batch = batch.loc[is_train[batch.file_row_number.to_numpy()]]
            if batch.empty: continue
            x = batch[self.numeric].to_numpy(dtype=float)
            neg += (x < 0).sum(axis=0);logged = self.log_values(x);n += len(x)
            for j, s in enumerate(scalers):
                values = logged[np.isfinite(logged[:,j]),j]
                if len(values): s.partial_fit(values[:,None])
            protocols.update(batch[self.protocol].astype(str).unique())
        mean = np.array([getattr(s,'mean_',[np.nan])[0] for s in scalers])
        var = np.array([getattr(s,'var_',[np.nan])[0] for s in scalers])
        scale = np.array([getattr(s,'scale_',[np.nan])[0] for s in scalers])
        active = np.isfinite(var) & (var > 0)
        if active.sum() < 2: raise ValueError('Insufficient variable features')
        p = {'numeric': self.numeric, 'active': active, 'mean': mean, 'scale': scale,
             'variance': var, 'protocols': sorted(protocols), 'n_train': n,
             'feature_names': list(np.array(self.numeric)[active]) + [f'Protocol={s}' for s in sorted(protocols)]}
        audit = pd.DataFrame({'feature': self.numeric, 'negative_to_missing_train': neg,
                              'log_mean': mean, 'log_std': np.sqrt(var), 'used': active})
        audit.to_csv(self.out / 'preprocessing_audit.csv', index=False)
        joblib.dump(p, artifact)
        return p

    @staticmethod
    def log_values(x):
        logged = np.full_like(x, np.nan)
        np.log1p(x, out=logged, where=np.isfinite(x) & (x >= 0))
        return logged

    def transform(self, frame, p):
        logged = self.log_values(frame[self.numeric].to_numpy(dtype=float))
        logged = np.where(np.isfinite(logged), logged, p['mean'])
        z = ((logged - p['mean']) / p['scale'])[:,p['active']]
        proto = frame[self.protocol].astype(str).to_numpy()
        encoded = (proto[:,None] == np.array(p['protocols'])[None,:]).astype('float32')
        x = np.concatenate([z.astype('float32'), encoded], axis=1)
        assert np.isfinite(x).all()
        return x, ~np.isin(proto, p['protocols'])

    def prepare_matrices(self):
        marker = self.out / 'matrices_complete.json'
        if marker.exists(): return read_json(marker)
        counts = self.freeze_splits();p = self.fit_preprocessor();d = len(p['feature_names'])
        stats = {}
        for year, splits in [('2017',['train_2017','test_2017']),('2018',['test_2018'])]:
            n_source = self.con.sql(f'SELECT count(*) FROM {self.source(year)}').fetchall()[0][0]
            lookups, arrays, hashes, unknowns = {}, {}, {}, {}
            for split in splits:
                rows = pd.read_parquet(self.out / f'{split}_rows.parquet', columns=['source_row'])
                lookup = np.full(n_source,-1,dtype=np.int32);lookup[rows.source_row.to_numpy()] = np.arange(len(rows))
                lookups[split] = lookup
                arrays[split] = np.lib.format.open_memmap(self.out / f'{split}_X.npy',mode='w+',dtype='float32',shape=(len(rows),d))
                hashes[split] = np.zeros(len(rows),dtype=np.uint64);unknowns[split] = np.zeros(len(rows),dtype=bool)
            for batch in self.batches(year):
                x, unknown = self.transform(batch,p)
                signature = pd.util.hash_pandas_object(pd.DataFrame(x),index=False).to_numpy(dtype=np.uint64)
                ids = batch.file_row_number.to_numpy()
                for split in splits:
                    pos = lookups[split][ids];mask=pos>=0
                    arrays[split][pos[mask]] = x[mask]
                    hashes[split][pos[mask]] = signature[mask];unknowns[split][pos[mask]] = unknown[mask]
            for split in splits:
                arrays[split].flush()
                np.save(self.out / f'{split}_feature_hash.npy',hashes[split])
                np.save(self.out / f'{split}_unknown_protocol.npy',unknowns[split])
                stats[split] = {'rows': len(arrays[split]), 'features': d, 'unknown_protocol': int(unknowns[split].sum())}
            del arrays,lookups
        train_hash = np.unique(np.load(self.out / 'train_2017_feature_hash.npy'))
        for split in ['test_2017','test_2018']:
            matched = np.isin(np.load(self.out / f'{split}_feature_hash.npy'),train_hash)
            np.save(self.out / f'{split}_training_hash_match.npy',matched)
            stats[split]['training_feature_hash_match_rate'] = float(matched.mean())
        write_json(marker,stats)
        return stats

    def fit_model(self, name):
        from .models import build_cleanlab_model
        target = self.out / f'{name}.joblib'
        if target.exists(): return read_json(self.out / f'{name}_fit.json')
        x = np.load(self.out / 'train_2017_X.npy',mmap_mode='r')
        y = pd.read_parquet(self.out / 'train_2017_rows.parquet',columns=['y']).y.to_numpy()
        if name=='logreg':
            # Factory includes a scaler; features are already train-standardized.
            model = build_cleanlab_model('logreg',2,self.seed).steps[-1][1]
            model.set_params(solver='saga',max_iter=150,tol=1e-3,random_state=self.seed)
        elif name=='rf':
            model = build_cleanlab_model('rf',2,self.seed)
            model.set_params(n_estimators=100,n_jobs=2,max_depth=20,max_leaf_nodes=4096,
                             min_samples_leaf=5,max_samples=min(200_000,len(y)))
        else: raise ValueError(name)
        start=time.time()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always');model.fit(x,y)
        info={'model':name,'training_rows':len(y),'features':x.shape[1], 'seconds':time.time()-start,
              'parameters':model.get_params(),'warnings':[str(w.message) for w in captured],
              'iterations':getattr(model,'n_iter_',np.array([])).tolist()}
        joblib.dump(model,target,compress=3);write_json(self.out / f'{name}_fit.json',info)
        return info

    def predict(self, name):
        model=joblib.load(self.out / f'{name}.joblib')
        for split in ['test_2017','test_2018']:
            path=self.out / f'{name}_{split}_predictions.parquet'
            if path.exists(): continue
            x=np.load(self.out / f'{split}_X.npy',mmap_mode='r')
            rows=pd.read_parquet(self.out / f'{split}_rows.parquet')
            scores=np.empty(len(rows),dtype='float64')
            positive=int(np.flatnonzero(model.classes_==1)[0])
            for start in range(0,len(rows),self.batch_size):
                stop=min(start+self.batch_size,len(rows));scores[start:stop]=model.predict_proba(x[start:stop])[:,positive]
            rows['score']=scores;rows['prediction']=(scores>=.5).astype('int8')
            rows['unknown_protocol']=np.load(self.out / f'{split}_unknown_protocol.npy')
            rows['training_feature_hash_match']=np.load(self.out / f'{split}_training_hash_match.npy')
            rows['split']=split;rows['model']=name
            rows.to_parquet(path,index=False)
        return self.model_reports(name)

    def model_reports(self,name):
        overall, labels=[],[]
        for split in ['test_2017','test_2018']:
            a=pd.read_parquet(self.out / f'{name}_{split}_predictions.parquet')
            for scope,mask in [('all',np.ones(len(a),bool)),('no_training_hash_match',~a.training_feature_hash_match)]:
                overall.append(dict(model=name,split=split,scope=scope,**binary_metrics(a.loc[mask])))
            train_labels=set(pd.read_parquet(self.out / 'train_2017_rows.parquet',columns=['label']).label)
            for label,g in a.groupby('label',observed=True):
                successes=int((g.prediction==g.y).sum());low,high=wilson(successes,len(g))
                labels.append(dict(model=name,split=split,label=label,rows=len(g),
                    metric='benign_false_positive_rate' if label=='BENIGN' else 'attack_recall',
                    value=float(g.prediction.mean()),ci_low=1-high if label=='BENIGN' else low,
                    ci_high=1-low if label=='BENIGN' else high,seen_label_in_training=label in train_labels))
        overall=pd.DataFrame(overall);labels=pd.DataFrame(labels)
        overall.to_csv(self.out / f'{name}_overall.csv',index=False);labels.to_csv(self.out / f'{name}_per_label.csv',index=False)
        return overall,labels


def wilson(success,n,z=1.96):
    if not n:return np.nan,np.nan
    p=success/n;denom=1+z*z/n;center=(p+z*z/(2*n))/denom
    half=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return max(0.0,min(p,center-half)),min(1.0,max(p,center+half))


def binary_metrics(frame):
    if frame.empty:return {'rows':0}
    y=frame.y.to_numpy();pred=frame.prediction.to_numpy();score=frame.score.to_numpy()
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    recall=tp/(tp+fn) if tp+fn else np.nan;fpr=fp/(fp+tn) if fp+tn else np.nan
    both=np.unique(y).size==2
    rl,rh=wilson(tp,tp+fn);fl,fh=wilson(fp,fp+tn)
    return dict(rows=len(y),benign_rows=int(tn+fp),attack_rows=int(tp+fn),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
        accuracy=float((tp+tn)/len(y)),attack_recall=recall,benign_fpr=fpr,
        recall_ci_low=rl,recall_ci_high=rh,fpr_ci_low=fl,fpr_ci_high=fh,
        precision=tp/(tp+fp) if tp+fp else np.nan,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan,
        balanced_accuracy=(recall+1-fpr)/2 if both else np.nan,
        roc_auc=roc_auc_score(y,score) if both else np.nan,
        average_precision=average_precision_score(y,score) if both else np.nan)


def coverage_directory(study):
    path=Path(getattr(study,'coverage_out',study.out / 'coverage'))
    path.mkdir(parents=True,exist_ok=True)
    return path


def fit_coverage(study, variance_target=.95):
    path=coverage_directory(study) / 'coverage_space.joblib'
    if path.exists():
        existing=joblib.load(path)
        if existing['variance_target'] != variance_target: raise ValueError('Cached PCA settings differ; use a new artifact directory')
        return existing
    p=joblib.load(study.out / 'preprocessor.joblib')
    active_names=list(np.array(study.numeric)[p['active']])
    indices=np.array([i for i,c in enumerate(active_names) if c in study.continuous])
    x=np.load(study.out / 'train_2017_X.npy',mmap_mode='r');m=Moments(len(indices))
    for start in range(0,len(x),study.batch_size):m.update(np.asarray(x[start:start+study.batch_size,indices],dtype=float))
    val,vec=np.linalg.eigh(m.m2/(m.n-1));order=np.argsort(val)[::-1];val=np.maximum(val[order],0);components=vec[:,order].T
    for component in components:
        if component[np.argmax(abs(component))]<0:component*=-1
    k=max(2,int(np.searchsorted(np.cumsum(val)/val.sum(),variance_target))+1)
    space={'indices':indices,'mean':m.mean,'components':components[:k],'eigenvalues':val,
           'n_pcs':k,'variance_retained':float(val[:k].sum()/val.sum()),'variance_target':variance_target}
    joblib.dump(space,path);return space


def project(x,space):
    return ((np.asarray(x[:,space['indices']],dtype=float)-space['mean'])@space['components'].T).astype('float32')


def coverage_reference(study,space,calibration_rows=3000,k_neighbors=5):
    marker=coverage_directory(study) / 'coverage_reference.json'
    if marker.exists():
        existing=read_json(marker)
        if existing['k_neighbors'] != k_neighbors or existing['calibration_rows'] != min(calibration_rows,existing['training_rows']):
            raise ValueError('Cached coverage reference settings differ; use a new artifact directory')
        return existing
    x=np.load(study.out / 'train_2017_X.npy',mmap_mode='r')
    scores=np.lib.format.open_memmap(coverage_directory(study) / 'training_pc_scores.npy',mode='w+',dtype='float32',shape=(len(x),space['n_pcs']))
    for start in range(0,len(x),study.batch_size):scores[start:start+study.batch_size]=project(x[start:start+study.batch_size],space)
    scores.flush()
    unique,inverse=np.unique(scores,axis=0,return_inverse=True)
    if len(unique)<=k_neighbors:raise ValueError('Too few unique coverage locations')
    np.save(coverage_directory(study) / 'unique_training_pc_scores.npy',unique)
    tree=KDTree(unique.astype('float64'),leaf_size=60)
    joblib.dump(tree,coverage_directory(study) / 'coverage_tree.joblib',compress=0)
    rng=np.random.default_rng(study.seed)
    idx=rng.choice(len(x),min(calibration_rows,len(x)),replace=False)
    # Unique reference locations avoid duplicate neighbors; remove the query's own site.
    distances,_=tree.query(unique[inverse[idx]].astype(float),k=k_neighbors+1)
    distance=distances[:,1:].mean(axis=1)
    q90,q99=np.quantile(distance,[.9,.99])
    if q99<=q90:raise ValueError('Degenerate coverage thresholds; inspect reference distances')
    pd.DataFrame({'train_matrix_row':idx,'mean_neighbor_distance':distance}).to_parquet(coverage_directory(study) / 'coverage_calibration.parquet',index=False)
    info={'training_rows':len(x),'unique_float32_pca_locations':len(unique),'merged_locations':len(x)-len(unique),
          'n_pcs':space['n_pcs'],'calibration_rows':len(idx),'k_neighbors':k_neighbors,
          'q90':float(q90),'q99':float(q99),'reference_policy':'All training locations, deduplicated in float32 PCA coordinates',
          'calibration_policy':'Uniform source-training rows; exclude each query site from reference neighbors'}
    write_json(marker,info);return info


def select_coverage_queries(study,population_per_split=15000,minimum_per_label=300):
    path=coverage_directory(study) / 'coverage_queries.parquet'
    settings={'population_per_split':population_per_split,'minimum_per_label':minimum_per_label}
    settings_path=coverage_directory(study) / 'coverage_query_settings.json'
    if path.exists():
        if not settings_path.exists() or read_json(settings_path)!=settings: raise ValueError('Cached query settings differ; use a new artifact directory')
        return pd.read_parquet(path)
    pieces=[];rng=np.random.default_rng(study.seed+100)
    for split in ['test_2017','test_2018']:
        rows=pd.read_parquet(study.out / f'{split}_rows.parquet')
        pop=rng.choice(len(rows),min(population_per_split,len(rows)),replace=False)
        population=np.zeros(len(rows),bool);population[pop]=True;supplement=np.zeros(len(rows),bool)
        for label,g in rows.groupby('label',observed=True):
            idx=g.index.to_numpy();needed=max(0,minimum_per_label-int(population[idx].sum()))
            candidates=idx[~population[idx]]
            if needed and len(candidates):supplement[rng.choice(candidates,min(needed,len(candidates)),replace=False)]=True
        chosen=rows.loc[population|supplement].copy();chosen['matrix_row']=chosen.index
        chosen['population_sample']=population[chosen.index];chosen['label_supplement']=supplement[chosen.index]
        chosen['split']=split;pieces.append(chosen)
    queries=pd.concat(pieces,ignore_index=True)
    assert queries.row_key.is_unique
    queries.to_parquet(path,index=False);write_json(settings_path,settings);return queries


def assess_coverage(study,space,reference,queries):
    path=coverage_directory(study) / 'coverage_query_distances.parquet'
    if path.exists():return pd.read_parquet(path)
    tree=joblib.load(coverage_directory(study) / 'coverage_tree.joblib');pieces=[]
    for split,g in queries.groupby('split',sort=False):
        x=np.load(study.out / f'{split}_X.npy',mmap_mode='r');frames=[]
        for start in range(0,len(g),study.batch_size):
            frame=g.iloc[start:start+study.batch_size].copy()
            z=project(x[frame.matrix_row.to_numpy()],space)
            distance,_=tree.query(z.astype(float),k=reference['k_neighbors'])
            frame['coverage_distance']=distance.mean(axis=1);frame['PC1']=z[:,0];frame['PC2']=z[:,1];frames.append(frame)
        pieces.append(pd.concat(frames,ignore_index=True))
    out=pd.concat(pieces,ignore_index=True)
    out['coverage_band']=pd.cut(out.coverage_distance,[-np.inf,reference['q90'],reference['q99'],np.inf],
                                labels=['supported','intermediate','low_support'],right=True).astype(str)
    out.to_parquet(path,index=False);return out


def coverage_reports(study,coverage,models=('logreg','rf')):
    summaries=[];label_rows=[]
    for model in models:
        predictions=pd.concat([pd.read_parquet(study.out / f'{model}_{split}_predictions.parquet')
                               for split in ['test_2017','test_2018']],ignore_index=True)
        # Coverage labels and matrix_row come from frozen split metadata, never inferred by ordering.
        joined=coverage.merge(predictions[['row_key','score','prediction','unknown_protocol','training_feature_hash_match']],on='row_key',validate='one_to_one')
        joined.to_parquet(coverage_directory(study) / f'{model}_coverage_predictions.parquet',index=False)
        population=joined[joined.population_sample]
        for scope,selected in [('all',population),('no_training_hash_match',population[~population.training_feature_hash_match])]:
            for (split,band),g in selected.groupby(['split','coverage_band'],observed=True):
                summaries.append(dict(model=model,scope=scope,split=split,coverage_band=band,**binary_metrics(g)))
        for (split,label,band),g in joined.groupby(['split','label','coverage_band'],observed=True):
            # Within-label supplementation is uniform among remaining rows; union is valid for per-label summaries.
            success=int(g.prediction.sum());lo,hi=wilson(success,len(g))
            label_rows.append(dict(model=model,split=split,label=label,coverage_band=band,rows=len(g),
                metric='benign_false_positive_rate' if label=='BENIGN' else 'attack_recall',
                value=success/len(g),ci_low=lo,ci_high=hi))
    summary=pd.DataFrame(summaries);per_label=pd.DataFrame(label_rows)
    summary.to_csv(coverage_directory(study) / 'coverage_performance.csv',index=False)
    per_label.to_csv(coverage_directory(study) / 'coverage_per_label.csv',index=False)
    return summary,per_label
