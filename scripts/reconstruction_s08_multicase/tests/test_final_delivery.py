"""Scope tests: a passing subset cannot qualify as the five-run delivery."""
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

path = Path(__file__).resolve().parents[1]/'final_delivery.py'
spec = importlib.util.spec_from_file_location('final_delivery', path)
delivery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(delivery)


class DeliveryScopeTests(unittest.TestCase):
    def setUp(self):
        self.keys = ['long','weak01','weak005','605','819']
        self.movies = self.keys+['s08_extension_comparison']
        self.v = dict(actual_visual_acceptance=True, boundary_reviewed=True, case_keys=self.keys.copy(),
            pdf_source_sha256={k:'a'*64 for k in delivery.PDFS}, movie_source_sha256={k:'b'*64 for k in self.movies},
            inspected_encoded_frames={k:[0,10] for k in self.movies}, assembly_sha256='c'*64)

    def test_complete_scope(self):
        delivery.visual_scope(self.v, self.keys, self.movies, 'c'*64)

    def test_single_case_cannot_qualify(self):
        self.v['case_keys'] = ['605']
        with self.assertRaises(AssertionError):delivery.visual_scope(self.v,self.keys,self.movies,'c'*64)

    def test_boundary_pdf_is_required(self):
        del self.v['pdf_source_sha256']['s08_extension_boundary_lapse.pdf']
        with self.assertRaises(AssertionError):delivery.visual_scope(self.v,self.keys,self.movies,'c'*64)

    def test_synchronized_movie_is_required(self):
        del self.v['movie_source_sha256']['s08_extension_comparison']
        with self.assertRaises(AssertionError):delivery.visual_scope(self.v,self.keys,self.movies,'c'*64)

    def test_assembly_change_invalidates_visual_binding(self):
        with self.assertRaises(AssertionError):delivery.visual_scope(self.v,self.keys,self.movies,'d'*64)

    def test_AH_tail_is_capped_at_requested_limit(self):
        c = dict(id='example', target=200)
        a = dict(numerical_checks_passed=True, checkpoint=dict(initial_count=2_000_000,dt=.025), first_accepted_AH_M=190, endpoint_M=200, case=dict(key='example'))
        self.assertEqual(delivery.endpoint(c,a),200)
        a['endpoint_M'] = 210
        with self.assertRaises(AssertionError):delivery.endpoint(c,a)

    def test_survival_requires_full_requested_cap(self):
        c = dict(id='example', target=300)
        a = dict(numerical_checks_passed=True, checkpoint=dict(initial_count=2_000_000,dt=.025), first_accepted_AH_M=None, endpoint_M=200, case=dict(key='example'))
        with self.assertRaises(AssertionError):delivery.endpoint(c,a)

    def test_probe_uses_movie_and_rejects_decoded_preview(self):
        response = '{"streams":[{"avg_frame_rate":"15/1","nb_frames":"178"}]}'
        movie = Path('/review/movies/case.mp4')
        with patch.object(delivery, 'run', return_value=response) as probe:
            delivery.probe_movie(movie, 178)
            self.assertEqual(probe.call_args.args[0][-1], movie)
            with self.assertRaises(AssertionError):
                delivery.probe_movie('/review/plots/case_decoded_0.png', 178)
            self.assertEqual(probe.call_count, 1)

    def test_probe_refuses_wrong_encoded_cadence_or_count(self):
        for response in ['{"streams":[{"avg_frame_rate":"30/1","nb_frames":"178"}]}',
                         '{"streams":[{"avg_frame_rate":"15/1","nb_frames":"177"}]}']:
            with patch.object(delivery, 'run', return_value=response):
                with self.assertRaises(AssertionError):delivery.probe_movie('/review/movies/case.mp4', 178)


if __name__ == '__main__':
    unittest.main()
