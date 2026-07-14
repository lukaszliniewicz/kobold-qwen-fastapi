from pathlib import Path

import run


def _configure_install(monkeypatch, tmp_path: Path, system: str):
    monkeypatch.setattr(run, "PROJECT_DIR", tmp_path)
    monkeypatch.setattr(run.platform, "system", lambda: system)


def test_legacy_upstream_binary_is_replaced(monkeypatch, tmp_path):
    _configure_install(monkeypatch, tmp_path, "Linux")
    binary = tmp_path / "bin" / "koboldcpp"
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"upstream")
    downloads = []

    def fake_download(url, destination, *, force=False):
        downloads.append((url, Path(destination), force))
        Path(destination).write_bytes(b"patched")

    monkeypatch.setattr(run, "download_file", fake_download)

    assert run.ensure_kobold_binary("vulkan") == binary
    assert downloads == [
        (run.KOBOLD_BASE_URL + "koboldcpp-linux-x64-nocuda", binary, True)
    ]
    assert binary.read_bytes() == b"patched"
    marker = (tmp_path / "bin" / ".koboldcpp-release").read_text(encoding="utf-8")
    assert run.KOBOLD_BINARY_RELEASE in marker


def test_matching_patched_binary_is_reused(monkeypatch, tmp_path):
    _configure_install(monkeypatch, tmp_path, "Windows")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(parents=True)
    binary = bin_dir / "koboldcpp.exe"
    binary.write_bytes(b"patched")
    (bin_dir / ".koboldcpp-release").write_text(
        f"{run.KOBOLD_BINARY_RELEASE}\n{run.KOBOLD_BASE_URL}\n",
        encoding="utf-8",
    )
    downloads = []
    monkeypatch.setattr(run, "download_file", lambda *args, **kwargs: downloads.append((args, kwargs)))

    assert run.ensure_kobold_binary("cuda") == binary
    assert downloads == []


def test_cuda_linux_selects_cuda_asset(monkeypatch, tmp_path):
    _configure_install(monkeypatch, tmp_path, "Linux")
    downloads = []

    def fake_download(url, destination, *, force=False):
        downloads.append(url)
        Path(destination).write_bytes(b"patched")

    monkeypatch.setattr(run, "download_file", fake_download)

    run.ensure_kobold_binary("cuda")

    assert downloads == [run.KOBOLD_BASE_URL + "koboldcpp-linux-x64"]
