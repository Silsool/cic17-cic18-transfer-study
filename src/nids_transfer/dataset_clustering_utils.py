"""Memory-bounded preparation and pooled-year HDBSCAN experiments.

Source Parquet files are only read. Label/family/year are evaluation metadata.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import duckdb
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import HDBSCAN
from sklearn.metrics import (adjusted_rand_score, adjusted_mutual_info_score,
                             homogeneity_score, completeness_score)


def qi(name):
    return '"' + name.replace('"', '""') + '"'


def literal(value):
    return "'" + str(value).replace("'", "''") + "'"


def normalize(name):
    return re.sub('[^a-z0-9]', '', name.lower())


METADATA = {'id', 'flowid', 'index', 'unnamed0', 'srcip', 'dstip', 'sourceip',
            'destinationip', 'srcipaddress', 'dstipaddress', 'sourceipaddress',
            'destinationipaddress', 'timestamp', 'captureday', 'day', 'date',
            'label', 'attack', 'attacktype', 'attackcategory', 'attemptedcategory'}
CATEGORICAL = {'protocol', 'srcport', 'dstport', 'sourceport', 'destinationport',
               'icmpcode', 'icmptype'}
NUMERIC_TYPES = {'tinyint', 'smallint', 'integer', 'bigint', 'hugeint', 'utinyint',
                 'usmallint', 'uinteger', 'ubigint', 'uhugeint', 'float', 'double', 'decimal'}


def coarse_family(label):
    """Evaluation convenience only; broad family does not establish equivalence."""
    s = str(label).lower()
    if s == 'benign': return 'Benign'
    if s.startswith('ddos'): return 'DDoS'
    if s.startswith('dos '): return 'DoS'
    if s.startswith('botnet'): return 'Botnet'
    if s.startswith('ftp-'): return 'FTP brute force'
    if s.startswith('ssh-'): return 'SSH brute force'
    if s.startswith('web attack'): return 'Web attack'
    if s == 'portscan' or ('infiltration' in s and 'portscan' in s): return 'Port scan'
    if s.startswith('infiltration'): return 'Infiltration'
    if s.startswith('heartbleed'): return 'Heartbleed'
    return 'Unmapped: ' + str(label)


class Moments:
    def __init__(self, p):
        self.n = 0
        self.mean = np.zeros(p)
        self.m2 = np.zeros((p, p))

    def merge(self, other):
        if not other.n: return
        total = self.n + other.n
        delta = other.mean - self.mean
        self.m2 += other.m2 + np.outer(delta, delta) * (self.n * other.n / total)
        self.mean += delta * (other.n / total)
        self.n = total

    def update(self, x):
        if not len(x): return
        b = Moments(x.shape[1])
        b.n, b.mean = len(x), x.mean(axis=0)
        centered = x - b.mean
        b.m2 = centered.T @ centered
        self.merge(b)


class FlowStudy:
    def __init__(self, paths, batch_size=32768):
        self.paths = {str(k): Path(v).resolve() for k, v in paths.items()}
        self.batch_size = batch_size
        self.con = duckdb.connect(config={'memory_limit': '2GB', 'threads': 2})
        schemas = {}
        for year, path in self.paths.items():
            if not path.exists(): raise FileNotFoundError(path)
            r = self.con.read_parquet(str(path))
            schemas[year] = dict(zip(r.columns, r.types))
        first = next(iter(schemas.values()))
        assert all(list(s) == list(first) for s in schemas.values())
        self.numeric = [c for c in first if normalize(c) not in METADATA | CATEGORICAL
                        and all(s[c].id in NUMERIC_TYPES for s in schemas.values())]
        self.flags = [c for c in self.numeric if 'flag' in normalize(c)]
        self.features = [c for c in self.numeric if c not in self.flags]
        self.label = next(c for c in first if normalize(c) == 'label')
        self.protocol = next(c for c in first if normalize(c) == 'protocol')
        # Same finite-row cleaning as the comparison notebook, including flags.
        self.valid = ' AND '.join(f'coalesce(isfinite({qi(c)}), false)' for c in self.numeric)

    def source(self, year):
        return f'read_parquet({literal(self.paths[year])}, file_row_number=true)'

    def where(self, label=None):
        return self.valid + (f' AND {qi(self.label)} = {literal(label)}' if label is not None else '')

    def inventory(self):
        rows = []
        for year in self.paths:
            data = self.con.sql(f'''SELECT {qi(self.label)} AS label,
                count(*) AS source_rows, count(*) FILTER (WHERE {self.valid}) AS clean_rows
                FROM {self.source(year)} GROUP BY 1 ORDER BY 1''').df()
            data.insert(0, 'year', year)
            rows.append(data)
        out = pd.concat(rows, ignore_index=True)
        out['family'] = out.label.map(coarse_family)
        out['attempted_variant'] = out.label.str.endswith(' - Attempted')
        return out

    def batches(self, year, label=None):
        selected = ', '.join(map(qi, self.features))
        rel = self.con.sql(f'SELECT {selected} FROM {self.source(year)} WHERE {self.where(label)}')
        reader = (rel.to_arrow_reader(batch_size=self.batch_size) if hasattr(rel, 'to_arrow_reader')
                  else rel.fetch_record_batch(self.batch_size))
        for batch in reader:
            yield batch.to_pandas().to_numpy(dtype=np.float64)

    @staticmethod
    def log_values(x):
        out = np.full_like(x, np.nan)
        np.log1p(x, out=out, where=np.isfinite(x) & (x >= 0))
        return out

    def fit_space(self, label=None):
        """Fit full-population log scalers and exact covariance PCA in two passes."""
        feature_scalers = [StandardScaler() for _ in self.features]
        audit, counts = [], {}
        for year in self.paths:
            negative = np.zeros(len(self.features), dtype=np.int64)
            n = 0
            for x in self.batches(year, label):
                n += len(x)
                negative += (x < 0).sum(axis=0)
                logged = self.log_values(x)
                for j, fitted in enumerate(feature_scalers):
                    observed = logged[np.isfinite(logged[:, j]), j]
                    if len(observed): fitted.partial_fit(observed[:, None])
            counts[year] = n
            audit.extend({'year': year, 'feature': c, 'negative_to_missing': int(negative[j]), 'rows': n}
                         for j, c in enumerate(self.features))
        if not all(counts.values()): raise ValueError(f'Empty year for label {label}: {counts}')
        scaler = StandardScaler()
        scaler.n_features_in_ = len(self.features)
        scaler.mean_ = np.array([getattr(s, 'mean_', [np.nan])[0] for s in feature_scalers])
        scaler.var_ = np.array([getattr(s, 'var_', [np.nan])[0] for s in feature_scalers])
        scaler.scale_ = np.array([getattr(s, 'scale_', [np.nan])[0] for s in feature_scalers])
        scaler.n_samples_seen_ = np.array([getattr(s, 'n_samples_seen_', 0) for s in feature_scalers])
        active = np.isfinite(scaler.var_) & (scaler.var_ > 0)
        space = {'scaler': scaler, 'active': active, 'features': np.array(self.features)[active],
                 'counts': counts, 'label': label, 'audit': pd.DataFrame(audit)}
        moments = Moments(int(active.sum()))
        for year in self.paths:
            for x in self.batches(year, label): moments.update(self.standardize(x, space))
        eigenvalues, eigenvectors = np.linalg.eigh(moments.m2 / (moments.n - 1))
        order = np.argsort(eigenvalues)[::-1]
        values, components = np.maximum(eigenvalues[order], 0), eigenvectors[:, order].T
        for component in components:
            if component[np.argmax(abs(component))] < 0: component *= -1
        space.update(mean=moments.mean, eigenvalues=values, components=components,
                     ratio=values / values.sum())
        assert np.allclose(components @ components.T, np.eye(len(values)))
        return space

    def standardize(self, x, space):
        log = self.log_values(x)
        scaler = space['scaler']
        log = np.where(np.isnan(log), scaler.mean_[None, :], log)
        z = scaler.transform(log)[:, space['active']]
        assert np.isfinite(z).all()
        return z

    def scores(self, frame, space):
        z = self.standardize(frame[self.features].to_numpy(dtype=float), space)
        return (z - space['mean']) @ space['components'].T

    def sample(self, per_year=50000, seed=42, label=None):
        rows = []
        selected = ', '.join(map(qi, ['file_row_number'] + self.numeric + [self.protocol, self.label]))
        for year in self.paths:
            # Reservoir sampling occurs AFTER cleaning and optional label selection.
            query = f'''SELECT {selected} FROM (
                SELECT * FROM {self.source(year)} WHERE {self.where(label)}
                ) USING SAMPLE reservoir ({int(per_year)} ROWS) REPEATABLE ({int(seed)})'''
            data = self.con.sql(query).df()
            data.insert(0, 'year', year)
            rows.append(data)
        frame = pd.concat(rows, ignore_index=True).rename(columns={self.label: 'label'})
        frame['row_key'] = frame.year + ':' + frame.file_row_number.astype(str)
        assert frame.row_key.is_unique
        frame['family'] = frame.label.map(coarse_family)
        frame['attempted_variant'] = frame.label.str.endswith(' - Attempted')
        return frame


def fit_auxiliary(frame, study):
    # Genuine flag counts: log1p then block scale, not rare-event z-scoring.
    flags = frame[study.flags].to_numpy(dtype=float)
    negative = (flags < 0).sum(axis=0)
    logged = np.full_like(flags, np.nan)
    np.log1p(flags, out=logged, where=flags >= 0)
    means = np.array([np.mean(c[np.isfinite(c)]) if np.isfinite(c).any() else 0
                      for c in logged.T])
    logged = np.where(np.isfinite(logged), logged, means)
    active = np.var(logged, axis=0) > 0
    flag_var_sum = np.var(logged[:, active], axis=0).sum()
    protocols = frame[study.protocol].astype('string').fillna('<missing>').to_numpy(dtype=str)
    return {'flag_mean': means, 'flag_active': active, 'flag_variance_sum': flag_var_sum,
            'protocols': sorted(set(protocols)),
            'flag_audit': pd.DataFrame({'flag': study.flags, 'negative_to_missing': negative,
                                        'used': active})}


def representation(frame, study, space, aux, n_pcs=12, protocol_weight=0, flag_weight=0):
    if not 2 <= n_pcs <= len(space['eigenvalues']): raise ValueError('Invalid number of PCs')
    # Expected squared PCA-block distance between independent pooled rows is ~1.
    denominator = np.sqrt(2 * space['eigenvalues'][:n_pcs].sum())
    pcs = study.scores(frame, space)
    blocks = [pcs[:, :n_pcs] / denominator]
    if protocol_weight:
        proto = frame[study.protocol].astype('string').fillna('<missing>').to_numpy(dtype=str)
        onehot = (proto[:, None] == np.array(aux['protocols'])[None, :]).astype(float)
        blocks.append(onehot * np.sqrt(protocol_weight / 2))
    if flag_weight and aux['flag_active'].any():
        raw = frame[study.flags].to_numpy(dtype=float)
        logged = np.full_like(raw, np.nan)
        np.log1p(raw, out=logged, where=raw >= 0)
        flags = np.where(np.isfinite(logged), logged, aux['flag_mean']) - aux['flag_mean']
        blocks.append(flags[:, aux['flag_active']] * np.sqrt(flag_weight / (2 * aux['flag_variance_sum'])))
    x = np.concatenate(blocks, axis=1)
    assert np.isfinite(x).all()
    return x, pcs


def run_experiment(name, frame, study, space, aux, n_pcs=12, protocol_weight=0,
                   flag_weight=0, min_cluster_size=500, min_samples=30):
    x, pcs = representation(frame, study, space, aux, n_pcs, protocol_weight, flag_weight)
    model = HDBSCAN(min_cluster_size=min_cluster_size, min_samples=min_samples,
                    metric='euclidean', cluster_selection_method='eom', n_jobs=2,
                    copy=True)
    clusters = model.fit_predict(x)
    assignments = frame[['row_key', 'year', 'label', 'family', 'attempted_variant']].copy()
    assignments['cluster'] = clusters
    assignments['membership_strength'] = model.probabilities_
    assignments['PC1'], assignments['PC2'] = pcs[:, 0], pcs[:, 1]
    return {'name': name, 'assignments': assignments, 'model': model,
            'parameters': {'n_pcs': n_pcs, 'protocol_weight': protocol_weight,
                           'flag_weight': flag_weight, 'min_cluster_size': min_cluster_size,
                           'min_samples': min_samples},
            'reports': evaluate(assignments)}


def alignment_scores(frame, column):
    records = []
    for scope, mask in [('all labels', np.ones(len(frame), bool)),
                        ('attacks only', frame.family.ne('Benign').to_numpy())]:
        subset = frame.loc[mask & frame.cluster.ge(0).to_numpy()]
        all_rows = int(mask.sum())
        valid = len(subset) > 1 and subset[column].nunique() > 1 and subset.cluster.nunique() > 1
        records.append({'scope': scope, 'annotation': column, 'rows': all_rows,
                        'assigned_rows': len(subset), 'coverage': len(subset) / all_rows if all_rows else np.nan,
                        'homogeneity': homogeneity_score(subset[column], subset.cluster) if valid else np.nan,
                        'completeness': completeness_score(subset[column], subset.cluster) if valid else np.nan,
                        'ARI': adjusted_rand_score(subset[column], subset.cluster) if valid else np.nan,
                        'AMI': adjusted_mutual_info_score(subset[column], subset.cluster) if valid else np.nan})
    return pd.DataFrame(records)


def evaluate(a):
    assigned = a[a.cluster >= 0]
    noise = a.assign(noise=a.cluster.eq(-1)).groupby(['year', 'label'], observed=True).agg(
        rows=('cluster', 'size'), noise_rows=('noise', 'sum'), noise_rate=('noise', 'mean'))
    year_noise = a.assign(noise=a.cluster.eq(-1)).groupby('year').agg(
        rows=('cluster', 'size'), noise_rate=('noise', 'mean'))
    cy = pd.crosstab(assigned.cluster, assigned.year)
    cy = cy.reindex(columns=sorted(a.year.unique()), fill_value=0)
    proportions = a.year.value_counts(normalize=True).reindex(cy.columns)
    cy_fraction = cy.div(cy.sum(axis=1), axis=0)
    enrichment = cy_fraction.div(proportions, axis=1)
    # Compare year mixing conditional on annotation as well as globally.
    conditional = []
    for annotation in ['label', 'family']:
        if assigned.empty: continue
        observed = assigned.groupby(['cluster', annotation, 'year']).size().unstack('year', fill_value=0)
        observed = observed.reindex(columns=cy.columns, fill_value=0)
        baseline = a.groupby([annotation, 'year']).size().unstack('year', fill_value=0)
        baseline = baseline.reindex(columns=cy.columns, fill_value=0)
        baseline = baseline.div(baseline.sum(axis=1), axis=0)
        for (cluster, value), row in observed.iterrows():
            for year in cy.columns:
                expected = baseline.loc[value, year]
                share = row[year] / row.sum()
                conditional.append({'annotation': annotation, 'cluster': cluster, 'category': value,
                                    'year': year, 'cluster_category_rows': int(row.sum()),
                                    'year_rows': int(row[year]), 'year_share': share,
                                    'category_year_baseline': expected,
                                    'year_enrichment': share / expected if expected else np.nan})
    tables = {}
    for annotation in ['label', 'family']:
        count = pd.crosstab(assigned.cluster, assigned[annotation])
        tables[annotation + '_counts'] = count
        tables[annotation + '_cluster_composition'] = count.div(count.sum(axis=1), axis=0)
        # Denominator includes noise: columns sum to coverage, not necessarily 1.
        totals = a[annotation].value_counts().reindex(count.columns)
        tables[annotation + '_capture'] = count.div(totals, axis=1)
    # Presence alone can overstate sharing; also require a small count in both years.
    mixed_ids = cy.index[cy.ge(10).all(axis=1)]
    mixing = a.assign(assigned=a.cluster.ge(0), shared=a.cluster.isin(mixed_ids)).groupby('year').agg(
        sampled_rows=('cluster', 'size'), assigned_rows=('assigned', 'sum'),
        rows_in_shared_clusters=('shared', 'sum'))
    mixing['fraction_of_all_rows'] = mixing.rows_in_shared_clusters / mixing.sampled_rows
    mixing['fraction_of_assigned_rows'] = mixing.rows_in_shared_clusters / mixing.assigned_rows.replace(0, np.nan)
    overview = pd.DataFrame([{'rows': len(a), 'clusters': assigned.cluster.nunique(),
                              'noise_rate': a.cluster.eq(-1).mean(),
                              'mixed_year_clusters': int((cy.gt(0).sum(axis=1) == len(cy.columns)).sum()),
                              'mixed_year_clusters_min10': len(mixed_ids)}])
    return dict(overview=overview, mixing_by_year=mixing, noise_by_label=noise, noise_by_year=year_noise,
                cluster_year_counts=cy, cluster_year_fraction=cy_fraction,
                year_enrichment=enrichment, conditional_year=pd.DataFrame(conditional),
                alignment=pd.concat([alignment_scores(a, 'label'), alignment_scores(a, 'family')], ignore_index=True),
                **tables)


def stability(left, right):
    joined = left['assignments'][['row_key', 'cluster']].merge(
        right['assignments'][['row_key', 'cluster']], on='row_key', suffixes=('_left', '_right'), validate='one_to_one')
    common = joined[(joined.cluster_left >= 0) & (joined.cluster_right >= 0)]
    return {'left': left['name'], 'right': right['name'], 'common_rows': len(joined),
            'assigned_in_both': len(common),
            'noise_status_agreement': np.mean(joined.cluster_left.eq(-1) == joined.cluster_right.eq(-1)) if len(joined) else np.nan,
            'ARI_assigned_in_both': adjusted_rand_score(common.cluster_left, common.cluster_right) if len(common) > 1 else np.nan}


def plot_report(result, max_per_year=12000, seed=42):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    a, r = result['assignments'], result['reports']
    # Shuffle to reduce systematic overplotting; same category colors in both panels.
    ids = sorted(a.loc[a.cluster >= 0, 'cluster'].unique())
    cmap = plt.get_cmap('turbo')
    colors = {int(k): cmap(i / max(1, len(ids) - 1)) for i, k in enumerate(ids)}
    colors[-1] = (0.65, 0.65, 0.65, 0.25)
    years = sorted(a.year.unique())
    fig, axes = plt.subplots(1, len(years), figsize=(15, 5), sharex=True, sharey=True, squeeze=False)
    for ax, year in zip(axes[0], years):
        group = a[a.year == year]
        sampled = group.sample(min(max_per_year, len(group)), random_state=seed)
        ax.scatter(sampled.PC1, sampled.PC2, c=[colors[int(k)] for k in sampled.cluster], s=3, rasterized=True)
        ax.set_title(f'{year}: {len(group):,} sampled rows; noise {group.cluster.eq(-1).mean():.1%}')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.grid(alpha=0.2)
    fig.suptitle(result['name'] + ' — common PCA coordinates, colors = clusters; gray = noise')
    # A cluster legend remains readable even when many clusters are found.
    handles = [Line2D([], [], marker='o', linestyle='', color=colors[k], label=str(k), markersize=4)
               for k in [-1] + ids]
    if len(handles) <= 40:
        fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=min(12, len(handles)), title='Cluster (-1 = noise)')
    elif ids:
        from matplotlib.colors import ListedColormap, BoundaryNorm
        from matplotlib.cm import ScalarMappable
        palette = ListedColormap([colors[int(k)] for k in ids])
        norm = BoundaryNorm(np.arange(len(ids) + 1) - 0.5, len(ids))
        positions = np.unique(np.linspace(0, len(ids) - 1, 12).astype(int))
        bar = fig.colorbar(ScalarMappable(norm=norm, cmap=palette), ax=list(axes[0]),
                           ticks=positions, fraction=0.025, pad=0.02)
        bar.set_ticklabels([str(ids[i]) for i in positions])
        bar.set_label('Cluster ID (gray = noise)')
    plt.show()
    for key, title in [('family_cluster_composition', 'Family composition within each cluster'),
                       ('family_capture', 'Fraction of each family captured (noise included in denominator)'),
                       ('cluster_year_fraction', 'Year composition within each cluster')]:
        data = r[key]
        if data.empty: continue
        fig, ax = plt.subplots(figsize=(max(7, len(data.columns)*0.8), min(20, max(4, len(data)*0.25))), constrained_layout=True)
        im = ax.imshow(data.to_numpy(), aspect='auto', vmin=0, vmax=1, cmap='viridis')
        ax.set_xticks(range(len(data.columns)), data.columns, rotation=60, ha='right')
        ax.set_yticks(range(len(data)), data.index)
        ax.set_ylabel('Cluster'); ax.set_title(result['name'] + '\n' + title)
        fig.colorbar(im, ax=ax, label='Fraction'); plt.show()
