"""Print full competitive matrix."""
import json
with open('.danteforge/compete/matrix.json', encoding='utf-8') as f:
    m = json.load(f)

EXCLUDED = {10,12,18,40,41,42,58,69,87,88,94,99}

print('DIM      SNTNL  BBG  CapIQ  FSt  LSEG  Morn  AlSns  OpenBB  QLIb  VBT   LABEL')
print('=' * 145)
for d in m['dimensions']:
    dim_num = int(d['id'].split('_')[1])
    s = d['scores']
    self_s = s.get('self', 0)
    bbg = s.get('bloomberg', 0)
    cap = s.get('capiq', 0)
    fst = s.get('factset', 0)
    lsg = s.get('lseg', 0)
    mst = s.get('morningstar', 0)
    als = s.get('alphasense', 0)
    obb = s.get('openbb', 0)
    qli = s.get('qlib', 0)
    vbt = s.get('vectorbt', 0)
    label = d['label'][:50].encode('ascii', 'replace').decode('ascii')
    excl = ' [EXCL]' if dim_num in EXCLUDED else ''
    print(f"{d['id']:8} {self_s:>6}  {bbg:>3}  {cap:>5}  {fst:>3}  {lsg:>4}  {mst:>4}  {als:>5}  {obb:>6}  {qli:>4}  {vbt:>3}   {label}{excl}")

print('=' * 145)

eligible = [d for d in m['dimensions'] if int(d['id'].split('_')[1]) not in EXCLUDED and d['scores'].get('self', 0) > 0]
total_w = sum(d['weight'] for d in eligible)
wself = sum(d['scores']['self'] * d['weight'] for d in eligible)
wbbg  = sum(d['scores'].get('bloomberg',0) * d['weight'] for d in eligible)
wcap  = sum(d['scores'].get('capiq',0) * d['weight'] for d in eligible)
wfst  = sum(d['scores'].get('factset',0) * d['weight'] for d in eligible)
wlsg  = sum(d['scores'].get('lseg',0) * d['weight'] for d in eligible)
wmst  = sum(d['scores'].get('morningstar',0) * d['weight'] for d in eligible)
wals  = sum(d['scores'].get('alphasense',0) * d['weight'] for d in eligible)
wobb  = sum(d['scores'].get('openbb',0) * d['weight'] for d in eligible)
wqli  = sum(d['scores'].get('qlib',0) * d['weight'] for d in eligible)
wvbt  = sum(d['scores'].get('vectorbt',0) * d['weight'] for d in eligible)
print(f"WEIGHTED {wself/total_w:>6.2f}  {wbbg/total_w:>3.1f}  {wcap/total_w:>5.1f}  {wfst/total_w:>3.1f}  {wlsg/total_w:>4.1f}  {wmst/total_w:>4.1f}  {wals/total_w:>5.1f}  {wobb/total_w:>6.1f}  {wqli/total_w:>4.1f}  {wvbt/total_w:>3.1f}")
print(f"n={len(eligible)} eligible dims, total_weight={total_w:.1f}")

print("\n--- BOTTOM 10 (SENTINEL score, worst first) ---")
scorable = [d for d in m['dimensions'] if int(d['id'].split('_')[1]) not in EXCLUDED and d['scores'].get('self', 0) > 0]
scorable.sort(key=lambda d: d['scores']['self'])
for d in scorable[:10]:
    s = d['scores']
    gap = s.get('bloomberg',0) - s.get('self',0)
    print(f"  {d['id']}: SENTINEL={s['self']} BBG={s.get('bloomberg',0)} gap={gap:+d}  {d['label'][:60]}")
