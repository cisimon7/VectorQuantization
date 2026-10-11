import json
import tarfile

import numpy as np
import pytest
from PIL import Image
from scipy.io import savemat

from data_processing import download, prepare
from vqgan.data import ImageDataset
from vqgan.training import evaluate_reconstruction


def test_stratified_splits_are_reproducible_and_disjoint():
    labels = np.repeat(np.arange(3), [40, 73, 258])
    splits = prepare.stratified_split(labels, 0)
    again = prepare.stratified_split(labels, 0)
    other = prepare.stratified_split(labels, 1)
    assert sorted(np.concatenate(list(splits.values())).tolist()) == list(range(1, len(labels) + 1))
    for name, ids in splits.items():
        np.testing.assert_array_equal(ids, again[name])
        assert not np.array_equal(ids, other[name])
        assert set(labels[ids - 1].tolist()) == {0, 1, 2}
        for label in range(3):
            ideal = (labels == label).sum() * (0.8 if name == "train" else 0.1)
            assert abs((labels[ids - 1] == label).sum() - ideal) <= 1


def test_aspect_preserving_center_crop_and_rgb(tmp_path):
    pixels = np.zeros((16, 32, 3), dtype=np.uint8)
    pixels[:, :8, 0] = 255
    pixels[:, 8:24, 1] = 255
    pixels[:, 24:, 2] = 255
    path = tmp_path / "wide.png"
    Image.fromarray(pixels).save(path)
    cropped = prepare.preprocess_image(path, 16)
    assert cropped.shape == (3, 16, 16)
    assert cropped.dtype == np.uint8
    np.testing.assert_array_equal(cropped[1], np.full((16, 16), 255))
    grayscale = tmp_path / "gray.png"
    Image.new("L", (16, 24), 42).save(grayscale)
    result = prepare.preprocess_image(grayscale, 16)
    np.testing.assert_array_equal(result[0], result[1])
    np.testing.assert_array_equal(result[1], result[2])


def test_uint8_loader_normalization_augmentation_and_padded_evaluation(tmp_path):
    path = tmp_path / "images.npy"
    images = np.zeros((3, 3, 8, 8), dtype=np.uint8)
    images[1] = 255
    images[2] = 128
    np.save(path, images)
    dataset = ImageDataset(path)
    assert isinstance(dataset.images, np.memmap)
    batch = dataset.batch([0, 1])
    assert batch.dtype == np.float32
    np.testing.assert_array_equal(batch[0], -np.ones((3, 8, 8)))
    np.testing.assert_array_equal(batch[1], np.ones((3, 8, 8)))
    augmented = dataset.batch([0, 1, 2], rng=np.random.default_rng(0), horizontal_flip=True)
    assert augmented.shape == (3, 3, 8, 8)
    batches = list(dataset.evaluation_batches(2))
    assert [valid for _, valid in batches] == [2, 1]
    assert all(batch.shape == (2, 3, 8, 8) for batch, _ in batches)
    np.testing.assert_array_equal(dataset.images, images)


def test_evaluation_excludes_padding_and_matches_manual_metrics(model, tmp_path):
    import jax
    import jax.numpy as jnp

    path = tmp_path / "images.npy"
    images = np.random.default_rng(0).uniform(-1, 1, (3, 3, 8, 8)).astype(np.float32)
    np.save(path, images)
    dataset = ImageDataset(path)
    result = evaluate_reconstruction(model, dataset, batch_size=2)
    predicted = np.asarray(jax.vmap(model)(jnp.asarray(images)).reconstruction)
    np.testing.assert_allclose(result["l1"], np.mean(abs(predicted - images)), rtol=1e-5)
    np.testing.assert_allclose(result["mse"], np.mean((predicted - images) ** 2), rtol=1e-5)
    assert result["num_images"] == 3
    subset = evaluate_reconstruction(model, dataset, batch_size=2, max_batches=1)
    assert subset["num_images"] == 2


def test_prepare_verify_and_corruption_detection(tmp_path, monkeypatch):
    root = tmp_path / "flowers"
    raw = root / "raw"
    jpg = raw / "jpg"
    jpg.mkdir(parents=True)
    labels = np.repeat(np.arange(1, 4), 10)
    savemat(raw / "imagelabels.mat", {"labels": labels[None, :]})
    savemat(
        raw / "setid.mat",
        {
            "trnid": np.arange(1, 11)[None, :],
            "valid": np.arange(11, 21)[None, :],
            "tstid": np.arange(21, 31)[None, :],
        },
    )
    for image_id in range(1, 31):
        Image.new("RGB", (16, 24), (image_id, 30, 40)).save(jpg / f"image_{image_id:05d}.jpg")
    with tarfile.open(raw / "102flowers.tgz", "w") as archive:
        for path in sorted(jpg.iterdir()):
            archive.add(path, arcname=f"jpg/{path.name}")
    monkeypatch.setattr(download, "IMAGE_COUNT", 30)
    monkeypatch.setattr(prepare, "IMAGE_COUNT", 30)
    monkeypatch.setattr(prepare, "NUM_CLASSES", 3)
    monkeypatch.setattr(prepare, "check_sources", lambda path: None)
    destination = prepare.prepare(root, size=16)
    metadata = prepare.verify(destination)
    assert [metadata["splits"][split]["count"] for split in prepare.SPLITS] == [24, 3, 3]
    assert prepare.prepare(root, size=16) == destination
    rows = json.loads((destination / "train.json").read_text())
    assert all(row["official_split"] in prepare.SPLITS for row in rows)
    with (destination / "train.npy").open("r+b") as stream:
        stream.seek(-1, 2)
        stream.write(b"\xff")
    with pytest.raises(ValueError, match="checksum"):
        prepare.verify(destination)


def test_source_checksum_rejects_corruption(tmp_path, monkeypatch):
    path = tmp_path / "source"
    path.write_bytes(b"correct")
    monkeypatch.setattr(
        download, "SOURCES", {"source": {"bytes": 7, "sha256": download.sha256(path)}}
    )
    download.check_sources(tmp_path)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        download.check_sources(tmp_path)


def test_archive_rejects_non_image_members(tmp_path):
    path = tmp_path / "images.tar"
    with tarfile.open(path, "w") as archive:
        archive.addfile(tarfile.TarInfo("../../unexpected"))
    with tarfile.open(path) as archive:
        with pytest.raises(ValueError, match="Unexpected archive entry"):
            download.archive_members(archive)
