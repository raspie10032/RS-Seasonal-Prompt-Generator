"""Run the existing VRS association scalar on CUDA, without rewriting its rule.

Requires CUDA PyTorch and its NVRTC runtime. The external VRS source remains
read-only; its exact header bytes are supplied to NVRTC and hashed in receipts.
No CPU inference fallback and no persistent device cache.
"""
import ctypes as C
import hashlib
from pathlib import Path
import sys

HEADERS = ("core_platform.hpp", "evidence_scalar.hpp", "association_scalar.hpp")
SOURCE = r'''
#include "swegca_architecture/association_scalar.hpp"
using namespace swegca::architecture::kernel;
extern "C" __global__ void association(const long long* support,
    const long long* refute, unsigned char* status, unsigned char* reason, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const auto result = judge_association({(unsigned long)support[i],
                                          (unsigned long)refute[i]});
    status[i] = (unsigned char)result.status();
    reason[i] = (unsigned char)result.reason();
}
'''


def source_receipt(core):
    core = Path(core).resolve()
    return {name: hashlib.sha256((core / "swegca_architecture" / name).read_bytes()).hexdigest()
            for name in HEADERS}


def _function(lib, name, types):
    fn = getattr(lib, name)
    fn.argtypes = types
    fn.restype = C.c_int
    return fn


class AssociationCUDA:
    def __init__(self, core):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA PyTorch is required; CPU fallback is disabled")
        self.torch = torch
        self.module = C.c_void_p()
        candidates = sorted(Path(sys.prefix).glob("lib/python*/site-packages/nvidia/*/lib/libnvrtc.so*"))
        if not candidates:
            raise RuntimeError("NVRTC runtime not found in this Python environment")
        self.nvrtc_path = str(candidates[0])
        rtc = C.CDLL(self.nvrtc_path)
        self.driver = C.CDLL("libcuda.so.1")
        ptr = C.c_void_p
        self._launch = _function(self.driver, "cuLaunchKernel", [ptr] + [C.c_uint] * 7 + [ptr, C.POINTER(ptr), C.POINTER(ptr)])
        self._unload = _function(self.driver, "cuModuleUnload", [ptr])
        self._error = _function(self.driver, "cuGetErrorString", [C.c_int, C.POINTER(C.c_char_p)])
        create = _function(rtc, "nvrtcCreateProgram", [C.POINTER(ptr), C.c_char_p, C.c_char_p, C.c_int, C.POINTER(C.c_char_p), C.POINTER(C.c_char_p)])
        compile_ = _function(rtc, "nvrtcCompileProgram", [ptr, C.c_int, C.POINTER(C.c_char_p)])
        log_size = _function(rtc, "nvrtcGetProgramLogSize", [ptr, C.POINTER(C.c_size_t)])
        get_log = _function(rtc, "nvrtcGetProgramLog", [ptr, ptr])
        ptx_size = _function(rtc, "nvrtcGetPTXSize", [ptr, C.POINTER(C.c_size_t)])
        get_ptx = _function(rtc, "nvrtcGetPTX", [ptr, ptr])
        destroy = _function(rtc, "nvrtcDestroyProgram", [C.POINTER(ptr)])
        names = [f"swegca_architecture/{h}".encode() for h in HEADERS]
        headers = [(Path(core) / n.decode()).read_bytes() for n in names]
        self.headers_sha256 = {name: hashlib.sha256(data).hexdigest()
                               for name, data in zip(HEADERS, headers)}
        array = C.c_char_p * len(names)
        program = ptr()
        if create(C.byref(program), SOURCE.encode(), b"rs_vrs_association.cu", len(names), array(*headers), array(*names)):
            raise RuntimeError("NVRTC program creation failed")
        try:
            # Initialize PyTorch's primary context, which owns all tensor pointers.
            torch.cuda.init()
            torch.empty(1, device="cuda")
            major, minor = torch.cuda.get_device_capability()
            options = [b"--std=c++17", b"--device-as-default-execution-space",
                       f"--gpu-architecture=compute_{major}{minor}".encode()]
            code = compile_(program, len(options), (C.c_char_p * len(options))(*options))
            size = C.c_size_t()
            if code:
                log_size(program, C.byref(size))
                log = C.create_string_buffer(size.value)
                get_log(program, log)
                raise RuntimeError("VRS CUDA compilation failed: " + log.value.decode())
            if ptx_size(program, C.byref(size)):
                raise RuntimeError("NVRTC PTX size query failed")
            ptx = C.create_string_buffer(size.value)
            if get_ptx(program, ptx):
                raise RuntimeError("NVRTC PTX extraction failed")
            self.ptx_sha256 = hashlib.sha256(ptx.raw).hexdigest()
            load = _function(self.driver, "cuModuleLoadData", [C.POINTER(ptr), ptr])
            self._check(load(C.byref(self.module), ptx))
            self.kernel = ptr()
            get_fn = _function(self.driver, "cuModuleGetFunction", [C.POINTER(ptr), ptr, C.c_char_p])
            self._check(get_fn(C.byref(self.kernel), self.module, b"association"))
        finally:
            destroy(C.byref(program))

    def _check(self, result):
        if result:
            message = C.c_char_p()
            self._error(result, C.byref(message))
            raise RuntimeError(f"CUDA driver error {result}: {message.value!r}")

    def judge(self, support, refute):
        torch = self.torch
        if support.shape != refute.shape or support.numel() > 2**31 - 1:
            raise ValueError("Matching bounded association columns required")
        if not support.is_cuda or support.device != refute.device or support.device.index != torch.cuda.current_device():
            raise ValueError("Association columns must be on the current CUDA device")
        if support.dtype != torch.int64 or refute.dtype != torch.int64:
            raise ValueError("Association counts must be int64")
        if bool((support < 0).any()) or bool((refute < 0).any()):
            raise ValueError("Association counts cannot be negative")
        support, refute = support.contiguous(), refute.contiguous()
        status = torch.empty_like(support, dtype=torch.uint8)
        reason = torch.empty_like(status)
        n = support.numel()
        if n:
            values = [C.c_void_p(t.data_ptr()) for t in (support, refute, status, reason)] + [C.c_int(n)]
            args = (C.c_void_p * len(values))(*(C.cast(C.byref(v), C.c_void_p) for v in values))
            stream = C.c_void_p(torch.cuda.current_stream().cuda_stream)
            self._check(self._launch(self.kernel, (n + 255) // 256, 1, 1, 256, 1, 1, 0, stream, args, None))
            torch.cuda.current_stream().synchronize()
        return status, reason

    def close(self):
        if self.module:
            self.torch.cuda.current_stream().synchronize()
            self._check(self._unload(self.module))
            self.module = C.c_void_p()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
