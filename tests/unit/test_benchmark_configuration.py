from pathlib import Path

from benchmarks.benchmark_runner import build_parser, run_throughput


ROOT = Path(__file__).resolve().parents[2]


def test_throughput_benchmark_declares_max_model_len():
    source = (ROOT / "benchmarks" / "throughput_benchmark.py").read_text()

    assert 'parser.add_argument("--max-model-len"' in source
    assert "max_model_len=args.max_model_len" in source


def test_benchmark_runner_forwards_max_model_len(monkeypatch):
    args = build_parser().parse_args(
        ["--tests", "throughput", "--throughput-max-model-len", "128"]
    )
    captured = {}

    def fake_run(command, **_kwargs):
        captured["command"] = command
        return type("Completed", (), {"stdout": "", "stderr": "", "returncode": 0})()

    monkeypatch.setattr("benchmarks.benchmark_runner.subprocess.run", fake_run)
    run_throughput(args)

    assert "--max-model-len" in captured["command"]
    index = captured["command"].index("--max-model-len")
    assert captured["command"][index + 1] == "128"
