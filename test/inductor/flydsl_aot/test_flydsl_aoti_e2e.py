# Owner(s): ["module: inductor"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import torch
from torch.autograd import DeviceType
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available


if not runtime_available(minimum_version=(0, 2, 3)):
    raise unittest.SkipTest("FlyDSL is not available")

import flydsl.compiler as flyc
import flydsl.expr as fx
from caffe2.test.inductor.flydsl_aot.flydsl_test_kernels import (
    ELEMENTWISE_BLOCK,
    GEMM_K,
    GEMM_M,
    GEMM_N,
    RMS_EPS,
    RMS_N,
    gemm_launcher,
    relu_launcher,
    rms_norm_launcher,
)
from torch._higher_order_ops.flydsl_kernel_wrap import (
    flydsl_kernel_wrapper_functional,
)
from torch._inductor.utils import fresh_cache, run_and_get_cpp_code


RELU_SCALE = 0.75

RELOCATED_PACKAGE_LOADER_SCRIPT = """
import sys

import torch

package_path = sys.argv[1]
lhs = torch.arange(1024, device="cuda", dtype=torch.float32)
rhs = torch.arange(1024, device="cuda", dtype=torch.float32).flip(0)
loader = torch._C._aoti.AOTIModelPackageLoader(
    package_path,
    "model",
    False,
    1,
    -1,
)
actual = loader.run([lhs, rhs])[0]
torch.cuda.synchronize()
torch.testing.assert_close(actual, lhs + rhs)
"""


@flyc.kernel
def _vector_add_kernel(
    lhs: fx.Tensor,
    rhs: fx.Tensor,
    out: fx.Tensor,
    block_dim: fx.Constexpr[int],
):
    block = fx.block_idx.x
    thread = fx.thread_idx.x

    lhs = fx.rocdl.make_buffer_tensor(lhs)
    rhs = fx.rocdl.make_buffer_tensor(rhs)
    out = fx.rocdl.make_buffer_tensor(out)
    tiled_lhs = fx.slice(
        fx.logical_divide(lhs, fx.make_layout(block_dim, 1)),
        (None, block),
    )
    tiled_rhs = fx.slice(
        fx.logical_divide(rhs, fx.make_layout(block_dim, 1)),
        (None, block),
    )
    tiled_out = fx.slice(
        fx.logical_divide(out, fx.make_layout(block_dim, 1)),
        (None, block),
    )
    tiled_lhs = fx.logical_divide(tiled_lhs, fx.make_layout(1, 1))
    tiled_rhs = fx.logical_divide(tiled_rhs, fx.make_layout(1, 1))
    tiled_out = fx.logical_divide(tiled_out, fx.make_layout(1, 1))

    copy_atom = fx.make_copy_atom(fx.rocdl.BufferCopy(32), fx.Float32)
    lhs_register = fx.make_rmem_tensor(1, fx.Float32)
    rhs_register = fx.make_rmem_tensor(1, fx.Float32)
    out_register = fx.make_rmem_tensor(1, fx.Float32)
    fx.copy_atom_call(copy_atom, fx.slice(tiled_lhs, (None, thread)), lhs_register)
    fx.copy_atom_call(copy_atom, fx.slice(tiled_rhs, (None, thread)), rhs_register)
    value = fx.arith.addf(
        fx.memref_load_vec(lhs_register),
        fx.memref_load_vec(rhs_register),
    )
    fx.memref_store_vec(value, out_register)
    fx.copy_atom_call(copy_atom, out_register, fx.slice(tiled_out, (None, thread)))


@flyc.jit
def _vector_add_launcher(
    out: fx.Tensor,
    lhs: fx.Tensor,
    rhs: fx.Tensor,
    elements: fx.Int32,
    block_dim: fx.Constexpr[int],
):
    grid = (elements + block_dim - 1) // block_dim
    _vector_add_kernel(lhs, rhs, out, block_dim).launch(
        grid=(grid, 1, 1),
        block=(block_dim, 1, 1),
    )


@flyc.jit
def _two_stage_add_launcher(
    out: fx.Tensor,
    workspace: fx.Tensor,
    lhs: fx.Tensor,
    rhs: fx.Tensor,
    elements: fx.Int32,
    block_dim: fx.Constexpr[int],
):
    grid = (elements + block_dim - 1) // block_dim
    _vector_add_kernel(lhs, rhs, workspace, block_dim).launch(
        grid=(grid, 1, 1),
        block=(block_dim, 1, 1),
    )
    _vector_add_kernel(workspace, rhs, out, block_dim).launch(
        grid=(grid, 1, 1),
        block=(block_dim, 1, 1),
    )


_captured_vector_add = torch.library.wrap_flydsl(
    _vector_add_launcher,
    mutates_args={"out"},
)
_captured_two_stage_add = torch.library.wrap_flydsl(
    _two_stage_add_launcher,
    mutates_args={"out", "workspace"},
)

_captured_gemm = torch.library.wrap_flydsl(
    gemm_launcher,
    mutates_args={"out"},
)
_captured_relu = torch.library.wrap_flydsl(
    relu_launcher,
    mutates_args={"out"},
)
_captured_rms_norm = torch.library.wrap_flydsl(
    rms_norm_launcher,
    mutates_args={"out"},
)


@torch.library.flydsl_op(
    "flydsl_aoti_test::two_stage_add",
    mutates_args=(),
)
def _two_stage_add(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    workspace = torch.empty_like(lhs)
    out = torch.empty_like(lhs)
    _captured_two_stage_add(
        out,
        workspace,
        lhs,
        rhs,
        lhs.numel(),
        ELEMENTWISE_BLOCK,
    )
    return out


class _VectorAddModel(torch.nn.Module):
    def forward(self, lhs, rhs):
        out = torch.empty_like(lhs)
        _captured_vector_add(
            out,
            lhs,
            rhs,
            lhs.numel(),
            256,
        )
        return out


class _TwoStageAddModel(torch.nn.Module):
    def forward(self, lhs, rhs):
        return torch.sin(_two_stage_add(lhs, rhs))


@torch.library.flydsl_op(
    "flydsl_aoti_test::composed",
    mutates_args=(),
)
def _composed(
    lhs: torch.Tensor,
    rhs: torch.Tensor,
    bias: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    gemm_out = torch.empty(
        (GEMM_M, GEMM_N),
        device=lhs.device,
        dtype=lhs.dtype,
    )
    _captured_gemm(lhs, rhs, gemm_out)

    biased = gemm_out + bias
    flat_biased = biased.flatten()
    flat_activated = torch.empty_like(flat_biased)
    _captured_relu(
        flat_biased,
        flat_activated,
        flat_biased.numel(),
        RELU_SCALE,
        ELEMENTWISE_BLOCK,
    )
    activated = flat_activated.view(GEMM_M, GEMM_N)

    mixed = activated + torch.sin(activated) * 0.25
    normalized = torch.empty_like(mixed)
    _captured_rms_norm(
        mixed,
        weight,
        normalized,
        GEMM_M,
        RMS_N,
    )
    final = normalized * 1.5 - 0.5
    return final, gemm_out, activated, normalized


class _ComposedModel(torch.nn.Module):
    def forward(self, lhs, rhs, bias, weight):
        return _composed(lhs, rhs, bias, weight)


class _DynamicRMSNormModel(torch.nn.Module):
    def forward(self, inp, weight):
        out = torch.empty_like(inp)
        _captured_rms_norm(
            inp,
            weight,
            out,
            inp.shape[0],
            RMS_N,
        )
        return out


class FlyDSLAOTIEndToEndTest(unittest.TestCase):
    def _assert_package_runs_after_relocation(
        self,
        package_path: str,
        root: Path,
    ) -> None:
        relocated_dir = root / "relocated"
        relocated_dir.mkdir()
        relocated_package = relocated_dir / "renamed_model.pt2"
        shutil.move(package_path, relocated_package)
        Path(package_path).parent.rmdir()

        child_cache = root / "child_cache"
        child_cache.mkdir()
        env = os.environ.copy()
        env["TORCHINDUCTOR_CACHE_DIR"] = str(child_cache)
        env["TRITON_CACHE_DIR"] = str(child_cache / "triton")
        env["PYTHONPATH"] = os.pathsep.join(path for path in sys.path if path)
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                RELOCATED_PACKAGE_LOADER_SCRIPT,
                str(relocated_package),
            ],
            cwd=relocated_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        self.assertEqual(
            0,
            result.returncode,
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        )

    def test_vector_add_runs_from_package(self):
        self.assertTrue(torch.cuda.is_available(), "ROCm GPU is not visible")
        self.assertIsNotNone(torch.version.hip, "PyTorch is not a ROCm build")
        lhs = torch.randn(1024, device="cuda", dtype=torch.float32)
        rhs = torch.randn_like(lhs)
        exported = torch.export.export(_VectorAddModel(), (lhs, rhs), strict=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            producer_dir = root / "producer"
            producer_dir.mkdir()
            with fresh_cache(dir=tmpdir):
                producer_cache = Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
                package_path, generated_code = run_and_get_cpp_code(
                    torch._inductor.aoti_compile_and_package,
                    exported,
                    package_path=str(producer_dir / "flydsl_vector_add.pt2"),
                    inductor_configs={"compile_threads": 1},
                )
            self.assertFalse(producer_cache.exists())
            self.assertNotIn("aoti_torch_clone", generated_code)
            self.assertNotIn("aoti_torch_copy_", generated_code)
            self.assertNotIn("triton_poi", generated_code)
            with zipfile.ZipFile(package_path) as package:
                packaged_files = package.namelist()
            self.assertTrue(
                any(path.endswith("/libfly_jit_runtime.so") for path in packaged_files),
                packaged_files,
            )
            self.assertTrue(
                any(
                    Path(path).name.startswith("libmlir_c_runner_utils.so")
                    for path in packaged_files
                ),
                packaged_files,
            )
            loader = torch._C._aoti.AOTIModelPackageLoader(
                package_path,
                "model",
                False,
                1,
                -1,
            )
            actual = loader.run([lhs, rhs])[0]
            torch.testing.assert_close(actual, lhs + rhs)

            stream = torch.cuda.Stream()
            self.assertNotEqual(
                stream.cuda_stream,
                torch.cuda.default_stream().cuda_stream,
            )
            marker = torch.empty_like(lhs)
            with torch.cuda.stream(stream):
                torch.add(lhs, rhs, out=marker)
                loader.run([lhs, rhs])
            torch.cuda.synchronize()

            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CUDA],
            ) as prof:
                with torch.cuda.stream(stream):
                    torch.add(lhs, rhs, out=marker)
                    stream_actual = loader.run([lhs, rhs])[0]
                torch.cuda.synchronize()
            gpu_events = [
                event for event in prof.events() if event.device_type == DeviceType.CUDA
            ]
            self.assertGreaterEqual(len(gpu_events), 2)
            self.assertEqual(
                1,
                len({event.device_resource_id for event in gpu_events}),
                [(event.name, event.device_resource_id) for event in gpu_events],
            )
            torch.testing.assert_close(stream_actual, lhs + rhs)

            del loader
            self._assert_package_runs_after_relocation(package_path, root)

    def test_flydsl_op_runs_multi_kernel_launcher_from_package(self):
        self.assertTrue(torch.cuda.is_available(), "ROCm GPU is not visible")
        self.assertIsNotNone(torch.version.hip, "PyTorch is not a ROCm build")
        lhs = torch.randn(1024, device="cuda", dtype=torch.float32)
        rhs = torch.randn_like(lhs)
        exported = torch.export.export(_TwoStageAddModel(), (lhs, rhs), strict=True)
        self.assertEqual(
            1,
            len(
                exported.graph_module.graph.find_nodes(
                    op="call_function",
                    target=torch.ops.flydsl_aoti_test.two_stage_add.default,
                )
            ),
        )
        decomposed = exported.run_decompositions(
            decompose_custom_flydsl_ops=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = torch._inductor.aoti_compile_and_package(
                decomposed,
                package_path=str(Path(tmpdir) / "flydsl_two_stage_add.pt2"),
                inductor_configs={"compile_threads": 1},
            )
            compiled = torch._inductor.aoti_load_package(package_path)
            actual = compiled(lhs, rhs)

        torch.testing.assert_close(actual, torch.sin(lhs + rhs + rhs))

    def test_dynamic_rms_norm_runs_multiple_row_counts(self):
        self.assertTrue(torch.cuda.is_available(), "ROCm GPU is not visible")
        self.assertIsNotNone(torch.version.hip, "PyTorch is not a ROCm build")
        inp = torch.randn(4, RMS_N, device="cuda", dtype=torch.float32)
        weight = torch.randn(RMS_N, device="cuda", dtype=torch.float32)
        rows = torch.export.Dim("rows", min=1, max=32)
        exported = torch.export.export(
            _DynamicRMSNormModel(),
            (inp, weight),
            dynamic_shapes=({0: rows}, None),
            strict=True,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(Path(tmpdir) / "flydsl_dynamic_rms_norm.pt2"),
                inductor_configs={"compile_threads": 1},
            )
            loader = torch._C._aoti.AOTIModelPackageLoader(
                package_path,
                "model",
                False,
                1,
                -1,
            )
            for row_count in (1, 4, 17, 32):
                test_inp = torch.randn(
                    row_count,
                    RMS_N,
                    device="cuda",
                    dtype=torch.float32,
                )
                actual = loader.run([test_inp, weight])[0]
                expected = (
                    test_inp
                    * torch.rsqrt(
                        test_inp.square().mean(dim=-1, keepdim=True) + RMS_EPS
                    )
                    * weight
                )
                torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)

    def test_composes_multiple_flydsl_and_pytorch_ops(self):
        self.assertTrue(torch.cuda.is_available(), "ROCm GPU is not visible")
        self.assertIsNotNone(torch.version.hip, "PyTorch is not a ROCm build")
        torch.manual_seed(0)
        lhs = torch.randn(GEMM_M, GEMM_K, device="cuda", dtype=torch.float32)
        rhs = torch.randn(GEMM_N, GEMM_K, device="cuda", dtype=torch.float32)
        bias = torch.randn(GEMM_N, device="cuda", dtype=torch.float32)
        weight = torch.randn(RMS_N, device="cuda", dtype=torch.float32)
        inputs = (lhs, rhs, bias, weight)
        exported = torch.export.export(_ComposedModel(), inputs, strict=True)
        flydsl_ops = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=torch.ops.flydsl_aoti_test.composed.default,
        )
        self.assertEqual(1, len(flydsl_ops))
        exported = exported.run_decompositions(decompose_custom_flydsl_ops=True)
        flydsl_nodes = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_functional,
        )
        self.assertEqual(3, len(flydsl_nodes))

        with tempfile.TemporaryDirectory() as tmpdir:
            package_path = torch._inductor.aoti_compile_and_package(
                exported,
                package_path=str(Path(tmpdir) / "flydsl_composed.pt2"),
                inductor_configs={"compile_threads": 1},
            )
            with zipfile.ZipFile(package_path) as package:
                packaged_files = package.namelist()
            self.assertEqual(
                1,
                sum(path.endswith("/libfly_jit_runtime.so") for path in packaged_files),
            )
            compiled = torch._inductor.aoti_load_package(package_path)
            actual = compiled(*inputs)

        expected_gemm = lhs @ rhs.T
        expected_activation = torch.relu(expected_gemm + bias) * RELU_SCALE
        mixed = expected_activation + torch.sin(expected_activation) * 0.25
        expected_normalized = (
            mixed
            * torch.rsqrt(mixed.square().mean(dim=-1, keepdim=True) + RMS_EPS)
            * weight
        )
        expected = (
            expected_normalized * 1.5 - 0.5,
            expected_gemm,
            expected_activation,
            expected_normalized,
        )
        for actual_value, expected_value in zip(actual, expected):
            torch.testing.assert_close(
                actual_value,
                expected_value,
                atol=2e-4,
                rtol=2e-4,
            )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
