"""Tests aot_compilation utils."""
from typing import cast

from axlearn.common import test_utils
from axlearn.common.aot_compilation import reshape_devices
from axlearn.common.utils import HybridMeshShape


class FakeDevice:
    pass


class AOTCompilationTest(test_utils.TestCase):
    def test_reshape_devices(self):
        devices = [FakeDevice()] * 8
        mesh_shape = (-1, 2)
        devices_per_slice = 8
        num_slices = 1

        devices, mesh_shape = reshape_devices(
            devices=devices,
            mesh_shape=mesh_shape,
            devices_per_slice=devices_per_slice,
            num_slices=num_slices,
        )
        self.assertEqual(mesh_shape, (4, 2))
        self.assertEqual(devices.shape, mesh_shape)

        devices = [FakeDevice()] * 8
        mesh_shape = HybridMeshShape(ici_mesh_shape=(1, -1), dcn_mesh_shape=(-1, 1))
        devices_per_slice = 4
        num_slices = 2
        devices, mesh_shape = reshape_devices(
            devices=devices,
            mesh_shape=mesh_shape,
            devices_per_slice=devices_per_slice,
            num_slices=num_slices,
        )
        self.assertIsInstance(mesh_shape, HybridMeshShape)
        mesh_shape = cast(HybridMeshShape, mesh_shape)  # Make pytype happy.
        self.assertEqual(mesh_shape.ici_mesh_shape, (1, 4))
        self.assertEqual(mesh_shape.dcn_mesh_shape, (2, 1))
        self.assertEqual(devices.shape, (2, 4))
