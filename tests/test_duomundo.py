import os
import shutil
import subprocess as sp


class TestCopamundial:
    @classmethod
    def setup_class(cls):
        cmd = "python setup.py install"
        proc = sp.Popen(cmd.split())
        proc.wait()
        os.makedirs("./tmp-copamundial-dir/", exist_ok = True)

    @classmethod
    def teardown_class(cls):
        shutil.rmtree("./tmp-copamundial-dir/")

    def _run_command(self, cmd):
        proc = sp.Popen(cmd.split())
        proc.wait()
        assert not proc.returncode

    def test_copamundial(self):
        cmd = "copamundial --ppiA ./netalign/tests/fly.s.tsv --ppiB ./netalign/tests/rat.s.tsv --nameA fly --nameB rat " \
              "--thres_dsd_dist 10 --dsd_A_dist ./tmp-copamundial-dir/fly-dsd-dist.npy --dsd_B_dist ./tmp-copamundial-dir/rat-dsd-dist.npy " \
              "--json_A ./tmp-copamundial-dir/fly.json --json_B ./tmp-copamundial-dir/rat.json --svd_AU ./tmp-copamundial-dir/fly-left-svd.npy " \
              "--svd_BU ./tmp-copamundial-dir/rat-left-svd.npy --svd_AV ./tmp-copamundial-dir/fly-right-svd.npy --svd_BV ./tmp-copamundial-dir/rat-right-svd.npy " \
              "--svd_r 100 --landmarks_a_b ./netalign/tests/fly-rat.tsv --compute_isorank " \
              "--model ./tmp-copamundial-dir/fly-rat.model --svd_dist_a_b ./tmp-copamundial-dir/fly-rat-svd-dist.npy --compute_go_eval --kA 10 --kB 10 " \
              "--metrics top-1-acc --output_file ./tmp-copamundial-dir/test-fly-rat.tsv --go_A ./data/go/fly.output.mapping.gaf " \
              "--go_B ./data/go/rat.output.mapping.gaf"
        self._run_command(cmd)
