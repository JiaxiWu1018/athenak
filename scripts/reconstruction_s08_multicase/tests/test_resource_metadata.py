"""A file renamed during du is transient; other inventory errors stay fatal."""
import importlib.util
from pathlib import Path
import subprocess
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('extension_resources',Path(__file__).resolve().parents[1]/'extension_resources.py')
resources=importlib.util.module_from_spec(spec)
spec.loader.exec_module(resources)


class StorageMetadataTests(unittest.TestCase):
    def result(self,status,stderr='',stdout='1024 /fixture\n'):
        return subprocess.CompletedProcess(['du'],status,stdout,stderr)

    def test_renamed_file_retries_without_using_partial_count(self):
        missing=self.result(1,"du: cannot access '/fixture/001.tmp.png': No such file or directory\n",'512 /fixture\n')
        with patch.object(resources.subprocess,'run',side_effect=[missing,self.result(0)]) as run, patch.object(resources.time,'sleep'):
            self.assertEqual(resources.allocated_bytes('/fixture'),1024)
            self.assertEqual(run.call_count,2)

    def test_permission_error_is_not_retried_or_accepted(self):
        with patch.object(resources.subprocess,'run',return_value=self.result(1,'du: cannot read directory: Permission denied\n')) as run:
            with self.assertRaises(subprocess.CalledProcessError):resources.allocated_bytes('/fixture')
            self.assertEqual(run.call_count,1)

    def test_persistent_missing_file_still_fails(self):
        with patch.object(resources.subprocess,'run',return_value=self.result(1,"du: cannot access '/fixture/gone': No such file or directory\n")) as run, patch.object(resources.time,'sleep'):
            with self.assertRaises(subprocess.CalledProcessError):resources.allocated_bytes('/fixture')
            self.assertEqual(run.call_count,5)


if __name__=='__main__':unittest.main()
