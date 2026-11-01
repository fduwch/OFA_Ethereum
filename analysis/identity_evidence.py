"""Exact-key evidence parsing and conservative resolution of external labels."""
from __future__ import annotations
from datetime import datetime, timezone
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import tomllib
from bs4 import BeautifulSoup

KEY = re.compile(r'0x[0-9a-f]{96}\Z')
EXTRACT_KEY = re.compile(r'0x[0-9a-fA-F]{96}(?![0-9a-fA-F])')
ALIASES = {
    'titan': 'titan', 'titan builder': 'titan',
    'beaverbuild': 'beaverbuild', 'beaverbuild.org': 'beaverbuild',
    'rsync': 'rsync', 'rsync-builder': 'rsync',
    'flashbots': 'flashbots', 'buildernet': 'buildernet',
    'nethermind': 'nethermind', 'btcs': 'btcs', 'builder+': 'btcs', 'btcs builder': 'btcs',
    'quasar': 'quasar', 'quasar builder': 'quasar',
    'bobthebuilder': 'bobthebuilder', 'bob the builder': 'bobthebuilder',
    'beelder': 'blockbeelder', 'blockbeelder': 'blockbeelder',
    'block beelder': 'blockbeelder', 'buildai': 'buildai',
    'bloxroute': 'bloxroute', 'boba-builder': 'boba',
    'jetbuilder': 'jetbuilder', 'jet builder': 'jetbuilder',
    'loki': 'loki', 'iobuilder': 'iobuilder', 'builder0x69': 'builder0x69',
    'eureka builder': 'eureka', 'eureka': 'eureka',
    'ethgas': 'ethgas', 'ethgas builder': 'ethgas', 'ethgas-realtime': 'ethgas',
    'antbuilder': 'antbuilder', 'blocknative': 'blocknative', 'eden': 'eden',
    'eigenphi': 'eigenphi', 'eth-builder': 'eth-builder', 'f1b.io': 'f1b',
    'gambit labs': 'gambit', 'lightspeedbuilder': 'lightspeed',
    'manifold': 'manifold', 'manta-builder': 'manta', 'manta builder': 'manta',
    'nfactorial': 'nfactorial', 'payload': 'payload', 'penguinbuild': 'penguin',
    'penguin': 'penguin', 'snailbuilder.sh': 'snailbuilder', 'tbuilder': 'tbuilder',
    'ultrasound': 'ultrasound', 'bombora.build': 'bombora',
    'bananabuild.org': 'bananabuild', 'shimmerblocks.com': 'shimmerblocks',
}
NON_ENTITY = {'imposter', 'default key', 'tbd', 'unknown', ''}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()


def parse_payload(text, parser):
    """Return (literal key, label, network, service); no prefix or tag matching."""
    if parser in ('titan', 'titan_html'):
        if parser == 'titan_html':
            soup = BeautifulSoup(text, 'html.parser')
            text = (soup.find('main') or soup).get_text('\n', strip=True)
        if 'Our builder will submit only using the following public keys' not in text:
            raise ValueError('Titan allowlist statement missing')
        return [(k, 'Titan', 'titan', 'titan') for k in sorted(set(EXTRACT_KEY.findall(text)))]
    if parser == 'buildernet':
        out = []
        for line in text.splitlines():
            if not line.startswith('|'):
                continue
            keys = EXTRACT_KEY.findall(line)
            if not keys:
                continue
            label = line.split('|')[1].strip().split(' (')[0]
            op = ALIASES.get(label.casefold())
            if op not in ('flashbots', 'nethermind', 'beaverbuild', None):
                raise ValueError('Unexpected BuilderNet operator')
            out.extend((k, label, 'buildernet', 'buildernet:'+op if op else '') for k in keys)
        return out
    if parser == 'brontes':
        data = tomllib.loads(text)
        pairs = [(k, b['name']) for b in data['builders'].values() for k in b.get('pub_keys', [])]
    elif parser == 'dataalways':
        pairs = [(r['pubkey'], r['builder_name']) for r in csv.DictReader(io.StringIO(text))]
    elif parser == 'yang':
        pairs = [(k, label) for label, keys in json.loads(text).items() for k in keys]
    elif parser == 'ethgas':
        pairs = [(k, b['name']) for b in json.loads(text)['data']['builderEntities'] for k in b['builderKeys']]
    else:
        return []
    out = []
    for key, label in pairs:
        name = ALIASES.get(label.strip().casefold(), '')
        out.append((key, label, name, name if name != 'buildernet' else ''))
    return out


def load_evidence(root):
    sources = []
    for folder in ('sources', 'registries_complete'):
        for meta in json.loads((root/folder/'manifest.json').read_text(encoding='utf-8-sig')):
            if 'file' not in meta:
                sources.append(meta)
                continue
            path = root/folder/meta['file']
            if sha256(path) != meta['sha256']:
                raise ValueError('Source checksum mismatch: '+str(path))
            sources.append(dict(meta, local_file=str(path.relative_to(root))))
    archive_index = root/'archive_transfer'/'titan_wayback_index.json'
    indexed = {x[0] for x in json.loads(archive_index.read_text())[1:]}
    for path in sorted((root/'archive_transfer').glob('titan_*.html')):
        stamp = path.stem.split('_')[1]
        if stamp not in indexed:
            raise ValueError('Unindexed archive timestamp')
        sources.append(dict(
            source_id='titan_archive_'+stamp, parser='titan_html',
            evidence_class='official_historical',
            published_at=datetime.strptime(stamp,'%Y%m%d%H%M%S').replace(tzinfo=timezone.utc).isoformat(),
            url=f'https://web.archive.org/web/{stamp}id_/https://docs.titanbuilder.xyz/builder-public-keys',
            local_file=str(path.relative_to(root)), sha256=sha256(path), bytes=path.stat().st_size,
            timestamp_kind='archive_capture',
        ))
    ledger = []
    for meta in sources:
        if 'local_file' not in meta or meta.get('parser','none') == 'none':
            continue
        records = parse_payload((root/meta['local_file']).read_text(encoding='utf-8-sig'), meta['parser'])
        seen = set()
        for key, label, network, service in records:
            key = key.strip().casefold()
            signature = (key, label)
            duplicate = signature in seen
            seen.add(signature)
            status = 'mapped'
            if not KEY.fullmatch(key):
                status = 'invalid_key'
            elif not network:
                status = 'non_entity_label' if label.casefold() in NON_ENTITY else 'unrecognized_label'
            ledger.append(dict(builder_pubkey=key, source_id=meta['source_id'], raw_label=label,
                               network=network, service=service, status=status, duplicate=duplicate,
                               published_at=meta.get('published_at',''),
                               evidence_class=meta['evidence_class'], url=meta['url']))
    return sources, ledger


def resolve_key(records, mode='historical', level='service'):
    """Official disclosures take precedence; registry disagreement stays unresolved.

    Mapping uses source labels only. The observed extra-data field is absent
    from this function. Distinct operating services stay separate from their
    network and from legacy services run by the same named operator.
    """
    valid = [r for r in records if r['status'] == 'mapped']
    if mode in ('historical','official_historical'):
        valid = [r for r in valid if r['evidence_class'].endswith('_historical')]
    elif mode not in ('all','official_all'):
        raise ValueError(mode)
    official = [r for r in valid if r['evidence_class'].startswith('official_')]
    if official:
        names = {r[level] for r in official if r[level]}
        networks = {r['network'] for r in official if r['network']}
        if len(names)>1 or len(networks)>1:
            return '', '', 'official_conflict', []
        if names:
            name = next(iter(names))
            support = [r for r in official if r[level] == name]
            dates = [r['published_at'] for r in support if r['published_at']]
            return name, min(dates) if dates else '', 'official', sorted({r['source_id'] for r in support})
        # Official network membership with unknown operator cannot be filled
        # using the observed tag or a generic network-only registry label.
        return '', '', 'network_only', sorted({r['source_id'] for r in official})
    if mode.startswith('official_'):
        return '', '', 'unmapped', []
    flags = [r for r in records if r['raw_label'].strip().casefold() in ('imposter','default key')]
    if mode == 'historical':
        flags = [r for r in flags if r['evidence_class'].endswith('_historical')]
    else:
        flags = [r for r in flags if r['evidence_class'].endswith('_current')]
    if flags:
        return '', '', 'registry_non_entity', sorted({r['source_id'] for r in flags})
    # Earlier versions contribute dates, not extra votes or superseded labels.
    regs = [r for r in valid if r['evidence_class'].startswith(('research_registry_','relay_registry_'))]
    if mode == 'all':
        updated_families = {r['source_id'].split('_')[0] for r in regs if r['evidence_class'].endswith('_current')}
        regs = [r for r in regs if r['source_id'].split('_')[0] not in updated_families or r['evidence_class'].endswith('_current')]
    names = {r[level] for r in regs if r[level]}
    if len(names)>1:
        return '', '', 'registry_conflict', sorted({r['source_id'] for r in regs})
    if not names:
        return '', '', 'unmapped', []
    name = next(iter(names))
    matching = [r for r in regs if r[level] == name]
    dates = [r['published_at'] for r in matching if r['published_at']]
    # Include earlier, matching snapshots for first documentary evidence.
    for r in records:
        if r['status']=='mapped' and r[level]==name and r['evidence_class'].endswith('_historical_earlier'):
            dates.append(r['published_at'])
    return name, min(dates) if dates else '', 'registry', sorted({r['source_id'] for r in matching})


def concentration_from_counts(counts):
    counts = sorted([int(x) for x in counts if x>0], reverse=True)
    n = sum(counts)
    if n==0:
        raise ValueError('Empty concentration sample')
    return dict(slots=n, identities=len(counts), hhi=sum(x*x for x in counts)/(n*n),
                cr3=sum(counts[:3])/n, cr5=sum(counts[:5])/n)
