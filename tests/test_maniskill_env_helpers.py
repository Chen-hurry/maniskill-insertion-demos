from pathlib import Path
import importlib.util

import numpy as np

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "test_maniskill_env.py"
SPEC = importlib.util.spec_from_file_location("test_maniskill_env_script", SCRIPT_PATH)
assert SPEC is not None
SCRIPT = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(SCRIPT)

episode_data_path = SCRIPT.episode_data_path
first_rgb_frame = SCRIPT.first_rgb_frame
is_done = SCRIPT.is_done
save_rgb_frames = SCRIPT.save_rgb_frames
save_episode_data = SCRIPT.save_episode_data
to_numpy_tree = SCRIPT.to_numpy_tree


def test_episode_data_path_uses_zero_padded_episode_id(tmp_path: Path) -> None:
    path = episode_data_path(tmp_path, 7)

    assert path == tmp_path / "episode_000007.npz"


def test_is_done_accepts_bool_and_tensor_like_values() -> None:
    assert is_done(True, False) is True
    assert is_done(False, True) is True
    assert is_done(np.array([False]), np.array([False])) is False
    assert is_done(np.array([False]), np.array([True])) is True


def test_to_numpy_tree_converts_nested_scalars_and_arrays() -> None:
    converted = to_numpy_tree(
        {
            "reward": 1.5,
            "state": [1, 2, 3],
            "info": {"success": np.array([True])},
        }
    )

    assert isinstance(converted["reward"], np.ndarray)
    assert converted["state"].tolist() == [1, 2, 3]
    assert converted["info"]["success"].tolist() == [True]


def test_save_episode_data_writes_compressed_npz(tmp_path: Path) -> None:
    path = save_episode_data(
        tmp_path,
        episode_id=3,
        observations=[{"state": np.array([0.0, 1.0])}],
        actions=[np.array([0.1, 0.2])],
        rewards=[1.0],
        dones=[False],
        infos=[{"success": np.array([False])}],
    )

    assert path.exists()
    loaded = np.load(path, allow_pickle=True)
    assert loaded["episode_id"].item() == 3
    assert loaded["num_steps"].item() == 1
    assert loaded["rewards"].tolist() == [1.0]


def test_first_rgb_frame_finds_nested_rgb_image() -> None:
    rgb = np.full((1, 8, 8, 3), 127, dtype=np.uint8)
    obs = {"sensor_data": {"base_camera": {"rgb": rgb}}}

    frame = first_rgb_frame(obs)

    assert frame is not None
    assert frame.shape == (8, 8, 3)
    assert frame.dtype == np.uint8


def test_first_rgb_frame_ignores_camera_matrices() -> None:
    obs = {
        "sensor_param": {"base_camera": {"intrinsic_cv": np.eye(3, dtype=np.float32)[None]}},
        "sensor_data": {"base_camera": {"rgb": np.zeros((1, 8, 8, 3), dtype=np.uint8)}},
    }

    frame = first_rgb_frame(obs)

    assert frame is not None
    assert frame.shape == (8, 8, 3)


def test_save_rgb_frames_writes_png_files(tmp_path: Path) -> None:
    frames = [
        np.zeros((4, 4, 3), dtype=np.uint8),
        np.full((4, 4, 3), 255, dtype=np.uint8),
    ]

    paths = save_rgb_frames(frames, tmp_path, episode_id=2)

    assert paths == [
        tmp_path / "rgb_frames" / "episode_000002" / "frame_000000.png",
        tmp_path / "rgb_frames" / "episode_000002" / "frame_000001.png",
    ]
    assert all(path.exists() for path in paths)
