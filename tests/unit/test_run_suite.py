import json

import run_suite
from benchmarks.benchmark_plot import write_throughput_svg


def _result(name):
    return {"test": name, "elapsed": 0.01, "returncode": 0, "output": "ok"}


def test_smoke_profile_selects_short_cpu_suite():
    args = run_suite.build_parser().parse_args(["--profile", "smoke"])

    assert run_suite._select_tests(args) == ["fragmentation", "thread-safety", "throughput"]


def test_run_suite_writes_metadata_results_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(run_suite.benchmark_runner, "run_fragmentation", lambda _args: _result("fragmentation"))
    monkeypatch.setattr(run_suite.benchmark_runner, "run_thread_safety", lambda _args: _result("thread-safety"))
    monkeypatch.setattr(run_suite.benchmark_runner, "run_throughput", lambda _args: _result("throughput"))

    args = run_suite.build_parser().parse_args(
        ["--profile", "smoke", "--results-dir", str(tmp_path)]
    )
    assert run_suite.run_suite(args) == 0

    run_dir = next(tmp_path.iterdir())
    metadata = json.loads((run_dir / "meta.json").read_text())
    results = json.loads((run_dir / "results.json").read_text())
    assert metadata["profile"] == "smoke"
    assert metadata["git"]["commit"]
    assert metadata["arguments"]["benchmark"]["throughput_max_model_len"] == 2048
    assert {item["test"] for item in results["results"]} == {
        "fragmentation",
        "thread-safety",
        "throughput",
    }

    def fail_if_called(_args):
        raise AssertionError("resume reran a completed benchmark")

    monkeypatch.setattr(run_suite.benchmark_runner, "run_fragmentation", fail_if_called)
    monkeypatch.setattr(run_suite.benchmark_runner, "run_thread_safety", fail_if_called)
    monkeypatch.setattr(run_suite.benchmark_runner, "run_throughput", fail_if_called)
    resume_args = run_suite.build_parser().parse_args(["--resume", str(run_dir)])

    assert run_suite.run_suite(resume_args) == 0


def test_validate_result_rejects_missing_fields():
    try:
        run_suite.validate_result({"test": "throughput"})
    except ValueError as exc:
        assert "missing fields" in str(exc)
    else:
        raise AssertionError("invalid benchmark result was accepted")


def test_benchmark_plot_reads_saved_csv(tmp_path):
    csv_path = tmp_path / "throughput_results.csv"
    csv_path.write_text(
        "variant,max_parallel,tokens_per_second\n"
        "paged,5,13.97\ncontiguous,5,27.41\n"
    )
    output_path = tmp_path / "throughput.svg"

    write_throughput_svg(csv_path, output_path)

    assert output_path.exists()
    assert "Maximum concurrent sequences" in output_path.read_text()
