"""Small safety/contract tests for the handoff runner; no model training."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('delivery',Path(__file__).with_name('delivery.py'))
delivery=importlib.util.module_from_spec(spec);spec.loader.exec_module(delivery)

class DeliveryContract(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)/'source';self.root.mkdir()
        self.patch=patch.object(delivery,'ROOT',self.root);self.patch.start();self.addCleanup(self.patch.stop)
    def test_no_snapshot_over_project_root(self):
        with self.assertRaises(ValueError):delivery.prepare(self.root)
    def test_no_snapshot_inside_raw_or_private(self):
        for path in ['data/raw/run','submission_private/run']:
            with self.assertRaises(ValueError):delivery.prepare(self.root/path)
    def test_unrecognised_directory_not_overwritten(self):
        out=Path(self.temp.name)/'run';out.mkdir();(out/'user.txt').write_text('keep')
        with patch.object(delivery,'inventory',return_value={'files':[]}),patch.object(delivery,'sha',return_value='manifest'):
            with self.assertRaises(RuntimeError):delivery.prepare(out)
        self.assertEqual((out/'user.txt').read_text(),'keep')
    def test_missing_dependency_fails_before_launch(self):
        snapshot=Path(self.temp.name)/'snapshot';(snapshot/'data/processed').mkdir(parents=True)
        with patch.object(delivery,'child') as child:
            with self.assertRaises(RuntimeError):delivery.run_task(snapshot,'same-information',[])
            child.assert_not_called()
    def test_existing_wrong_data_not_overwritten(self):
        path=self.root/'data/raw/homeacf_tstsa.rda';path.parent.mkdir(parents=True);path.write_bytes(b'wrong')
        with patch('urllib.request.urlopen') as network:
            with self.assertRaises(RuntimeError):delivery.fetch()
            network.assert_not_called()
        self.assertEqual(path.read_bytes(),b'wrong')
    def test_manifest_mismatch_blocks_packaging(self):
        manifest={'files':[{'path':'missing.py','sha256':'not-a-match'}]}
        with patch.object(delivery,'inventory',return_value=manifest),contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(RuntimeError):delivery.audit()
    def test_failed_or_partial_results_cannot_be_promoted(self):
        for result in [{'status':'running'},{'failures':[{'replicate':2}]},{'n_requested':300,'n_effective':299},
             {'design':{'anchor_gate':{'arm':{'passed':False}}}}]:
            with self.assertRaises(RuntimeError):delivery.validate_result(result)
        delivery.validate_result({'n_requested':300,'n_effective':300,'failures':[]})

if __name__=='__main__':unittest.main()
