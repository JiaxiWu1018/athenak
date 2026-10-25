"""Conditional per-trajectory resource extrapolation; shared caps stay fixed."""
import math

GIB = 1024**3


def project(cases, own_bytes, spent_or_reserved_seconds):
    assert cases and own_bytes >= 0 and spent_or_reserved_seconds >= 0
    rates = [c['observed_native_seconds_per_M'] for c in cases
             if c.get('observed_native_seconds_per_M') is not None]
    assert rates and all(math.isfinite(r) and r > 0 for r in rates)
    fallback_rate = max(rates)
    keys = ['particle_bytes', 'cohort_bytes', 'checkpoint_bytes']
    slices = sorted({k for c in cases for k in c['slice_bytes']})
    maxima = {k: max(c[k] for c in cases) for k in keys}
    max_slices = {k: max(c['slice_bytes'].get(k, 0) for c in cases) for k in slices}
    assert min(*maxima.values(), *max_slices.values()) > 0
    rows = []
    for case in cases:
        c = dict(case)
        assert math.isfinite(c['remaining']) and c['remaining'] >= 0
        assert c['future_scientific_anchors'] >= 0
        rate = c.get('observed_native_seconds_per_M')
        c['rate_source'] = 'own_observed_maximum' if rate is not None else 'global_fallback_unobserved'
        c['native_seconds_per_M'] = fallback_rate if rate is None else rate
        c['size_fallback_fields'] = [k for k in keys if c[k] == 0]
        for k in keys:
            assert c[k] >= 0
            c[k] = c[k] or maxima[k]
        c['slice_bytes'] = dict(c['slice_bytes'])
        for k in slices:
            assert c['slice_bytes'].get(k, 0) >= 0
            if not c['slice_bytes'].get(k):
                c['size_fallback_fields'].append('slice_'+k)
                c['slice_bytes'][k] = max_slices[k]
        c['projected_remaining_native_node_hours'] = c['remaining']*c['native_seconds_per_M']/3600
        c['future_cadence_bytes'] = math.ceil(c['remaining']/2)*(c['particle_bytes']+sum(c['slice_bytes'].values()))
        c['future_cadence_bytes'] += math.ceil(c['remaining']/.5)*c['cohort_bytes']
        # Retain the original allowance of two extra recoveries per case.
        c['checkpoint_allowance_bytes'] = (c['future_scientific_anchors']+2)*c['checkpoint_bytes']
        rows.append(c)
    # Retain one additional maximum-sized restart-identity checkpoint allowance.
    extra_identity = max(c['checkpoint_bytes'] for c in rows)
    cadence = sum(c['future_cadence_bytes'] for c in rows)
    checkpoints = sum(c['checkpoint_allowance_bytes'] for c in rows)+extra_identity
    native_hours = sum(c['projected_remaining_native_node_hours'] for c in rows)
    projected_bytes = own_bytes+cadence+checkpoints+20*GIB
    cost = spent_or_reserved_seconds/3600+native_hours+4
    return dict(cases=rows, projection_method='per_case_observed_maxima',
        remaining_long_evolution_M=sum(c['remaining'] for c in rows),
        future_scientific_anchors=sum(c['future_scientific_anchors'] for c in rows),
        future_cadence_bytes=cadence, checkpoint_allowance_bytes=checkpoints,
        extra_restart_identity_allowance_bytes=extra_identity,
        product_and_overhead_allowance_bytes=20*GIB,
        conservative_projected_new_bytes=projected_bytes,
        native_seconds_per_M=fallback_rate,
        native_seconds_per_M_definition='Maximum observed case rate; total cost uses each case separately',
        projected_remaining_native_node_hours=native_hours,
        spent_or_reserved_node_hours=spent_or_reserved_seconds/3600,
        cost_overhead_allowance_node_hours=4,
        conditional_projected_total_node_hours=cost,
        cost_projection_within_cap=bool(cost <= 50),
        storage_projection_within_cap=bool(projected_bytes <= 350*GIB))


def analytic_checks():
    cases = [dict(case='outward', remaining=100., future_scientific_anchors=1,
        observed_native_seconds_per_M=120., particle_bytes=10, cohort_bytes=2,
        checkpoint_bytes=100, slice_bytes=dict(a=1)),
        dict(case='control', remaining=200., future_scientific_anchors=2,
        observed_native_seconds_per_M=30., particle_bytes=20, cohort_bytes=3,
        checkpoint_bytes=40, slice_bytes=dict(a=2))]
    p = project(cases, 1000, 3600)
    assert p['projected_remaining_native_node_hours'] == 5
    assert p['conditional_projected_total_node_hours'] == 10
    assert p['future_cadence_bytes'] == 4350 and p['checkpoint_allowance_bytes'] == 560
    assert p['conservative_projected_new_bytes'] == 1000+4350+560+20*GIB
    fallback = [cases[0], dict(cases[1], observed_native_seconds_per_M=None,
        particle_bytes=0, cohort_bytes=0, checkpoint_bytes=0, slice_bytes=dict(a=0))]
    p = project(fallback, 1000, 0)
    assert p['cases'][1]['rate_source'] == 'global_fallback_unobserved'
    assert p['cases'][1]['native_seconds_per_M'] == 120 and p['cases'][1]['checkpoint_bytes'] == 100
    assert not project(cases, 350*GIB, 50*3600)['cost_projection_within_cap']
    assert not project(cases, 350*GIB, 0)['storage_projection_within_cap']
    completed = [dict(c, remaining=0., future_scientific_anchors=0) for c in cases]
    assert project(completed, 0, 0)['projected_remaining_native_node_hours'] == 0
    return dict(per_case_weighted_cost=True, per_case_cadence_and_checkpoints=True,
        unobserved_case_global_fallback=True, shared_caps_unchanged=True,
        completed_cases_no_future_evolution=True)
