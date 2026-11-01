import unittest
import numpy as np
import pandas as pd
from identity_evidence import KEY,parse_payload,resolve_key,concentration_from_counts
from analyze_identity import attach_variant

K1='0x'+'a'*96
K2='0x'+'b'*96


def evidence(key=K1, name='titan', cls='research_registry_historical',sid='test_historical',date='2024-01-01T00:00:00Z',service=None):
    return dict(builder_pubkey=key,network=name,service=name if service is None else service,
                raw_label=name,status='mapped',evidence_class=cls,source_id=sid,published_at=date)


class Tests(unittest.TestCase):
    def test_full_key_only(self):
        self.assertTrue(KEY.fullmatch(K1) is not None)
        for bad in [K1[:10],K1+'0','f'+K1,'0x'+'z'*96]:
            self.assertIsNone(KEY.fullmatch(bad))

    def test_titan_allowlist(self):
        text='Our builder will submit only using the following public keys:\n'+K1+'\n'+K1
        self.assertEqual(len(parse_payload(text,'titan')),1)
        with self.assertRaises(ValueError):parse_payload(K1,'titan')

    def test_titan_html_ignores_script(self):
        text='<main>Our builder will submit only using the following public keys:<code>'+K1+'</code></main><script>'+K2+'</script>'
        self.assertEqual([r[0] for r in parse_payload(text,'titan_html')],[K1])

    def test_buildernet_operator(self):
        text='| Flashbots (regular) | `'+K1+'` |\n| TBD (optimistic) | `'+K2+'` |'
        r=parse_payload(text,'buildernet')
        self.assertEqual(r[0][2:],('buildernet','buildernet:flashbots'))
        self.assertEqual(r[1][2:],('buildernet',''))

    def test_address_not_public_key(self):
        r=parse_payload('[builders."0x'+'a'*40+'"]\nname="titan"\npub_keys=["'+K1+'"]','brontes')
        self.assertEqual([x[0] for x in r],[K1])

    def test_unrecognized_not_guessed(self):
        r=parse_payload('pubkey,builder_name\n'+K1+',some titan-looking label','dataalways')
        self.assertEqual(r[0][2:],('',''))

    def test_registry_conflict_stays_separate(self):
        name,_,basis,_=resolve_key([evidence(),evidence(name='beaverbuild')])
        self.assertEqual(name,'');self.assertEqual(basis,'registry_conflict')

    def test_official_priority(self):
        r=evidence(name='buildernet',service='buildernet:flashbots',cls='official_historical')
        name,_,basis,_=resolve_key([r,evidence(name='flashbots')])
        self.assertEqual(name,'buildernet:flashbots');self.assertEqual(basis,'official')

    def test_official_conflict(self):
        r=[evidence(cls='official_historical'),evidence(name='beaverbuild',cls='official_historical')]
        self.assertEqual(resolve_key(r)[2],'official_conflict')

    def test_current_not_in_historical(self):
        r=[evidence(cls='official_current',date='')]
        self.assertEqual(resolve_key(r)[0],'')
        self.assertEqual(resolve_key(r,'all')[0],'titan')

    def test_updated_source_supersedes_old(self):
        r=[evidence(sid='reg_historical'),evidence(name='beaverbuild',cls='research_registry_current',sid='reg_current')]
        self.assertEqual(resolve_key(r,'historical')[0],'titan')
        self.assertEqual(resolve_key(r,'all')[0],'beaverbuild')

    def test_imposter_not_single_entity(self):
        flag=evidence();flag.update(status='non_entity_label',raw_label='imposter',network='',service='')
        self.assertEqual(resolve_key([flag,evidence()])[2],'registry_non_entity')

    def test_earlier_tbd_not_conflict(self):
        r=[evidence(name='buildernet',service='',cls='official_historical'),evidence(name='buildernet',service='buildernet:beaverbuild',cls='official_historical',date='2024-12-01T00:00:00Z')]
        self.assertEqual(resolve_key(r)[1],'2024-12-01T00:00:00Z')

    def test_distinct_unknown_keys(self):
        f=pd.DataFrame(dict(builder_pubkey=[K1,K2],slot_start_date_time=pd.to_datetime(['2024-01-01','2024-01-01'],utc=True)))
        m=pd.DataFrame(dict(builder_pubkey=[K1,K2],variant=['x','x'],name=['',''],documentary_date=['','']))
        ids,known=attach_variant(f,m,'x')
        self.assertEqual(len(set(ids)),2);self.assertFalse(known.any())

    def test_forward_time(self):
        f=pd.DataFrame(dict(builder_pubkey=[K1,K1],slot_start_date_time=pd.to_datetime(['2024-01-01','2024-02-01'],utc=True)))
        m=pd.DataFrame(dict(builder_pubkey=[K1],variant=['x'],name=['titan'],documentary_date=['2024-01-15T00:00:00Z']))
        ids,known=attach_variant(f,m,'x',True)
        np.testing.assert_array_equal(known,[False,True])
        self.assertNotEqual(ids[0],ids[1])

    def test_metrics_and_coarsening(self):
        raw=concentration_from_counts([3,2,1,1])
        merged=concentration_from_counts([5,1,1])
        self.assertAlmostEqual(raw['hhi'],15/49)
        self.assertAlmostEqual(merged['cr3'],1)
        self.assertGreaterEqual(merged['hhi'],raw['hhi'])
        with self.assertRaises(ValueError):concentration_from_counts([])

    def test_mixed_iso_dates(self):
        key3='0x'+'c'*96
        f=pd.DataFrame(dict(builder_pubkey=[K1,K2,key3],slot_start_date_time=pd.to_datetime(['2025-01-01']*3,utc=True)))
        m=pd.DataFrame(dict(builder_pubkey=[K1,K2,key3],variant=['x']*3,name=['titan']*3,
                            documentary_date=['2024-01-01T00:00:00Z','2024-01-01T00:00:00+00:00','2024-01-01T00:00:00.0000000+00:00']))
        _,known=attach_variant(f,m,'x',True)
        self.assertTrue(known.all())
        m.loc[1,'documentary_date']='invalid-date'
        with self.assertRaises(ValueError):attach_variant(f,m,'x',True)


if __name__=='__main__':unittest.main(verbosity=2)
