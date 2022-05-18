package cublas

// Copyright (c) 2018 Bhojpur Consulting Private Limited, India. All rights reserved.

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// #include <cublas_v2.h>
import "C"

// Status is the cublas status.
type Status int

func (err Status) Error() string  { return err.String() }
func (err Status) String() string { return resString[err] }

func status(x C.cublasStatus_t) error {
	err := Status(x)
	if err == Success {
		return nil
	}
	if err > LicenceError {
		return Unsupported
	}
	return err
}

const (
	Success        Status = C.CUBLAS_STATUS_SUCCESS          // The operation completed successfully.
	NotInitialized Status = C.CUBLAS_STATUS_NOT_INITIALIZED  // The cuBLAS library was not initialized. This is usually caused by the lack of a prior cublasCreate() call,
	AllocFailed    Status = C.CUBLAS_STATUS_ALLOC_FAILED     // Resource allocation failed inside the cuBLAS library.
	InvalidValue   Status = C.CUBLAS_STATUS_INVALID_VALUE    // An unsupported value or parameter was passed to the function (a negative vector size, for example).
	ArchMismatch   Status = C.CUBLAS_STATUS_ARCH_MISMATCH    // The function requires a feature absent from the device architecture; usually caused by the lack of support for double precision.
	MappingError   Status = C.CUBLAS_STATUS_MAPPING_ERROR    // An access to GPU memory space failed, which is usually caused by a failure to bind a texture.
	ExecFailed     Status = C.CUBLAS_STATUS_EXECUTION_FAILED // The GPU program failed to execute. This is often caused by a launch failure of the kernel on the GPU, which can be caused by multiple reasons.
	InternalError  Status = C.CUBLAS_STATUS_INTERNAL_ERROR   // An internal cuBLAS operation failed. This error is usually caused by a cudaMemcpyAsync() failure.
	Unsupported    Status = C.CUBLAS_STATUS_NOT_SUPPORTED    // The functionnality requested is not supported
	LicenceError   Status = C.CUBLAS_STATUS_LICENSE_ERROR    // The functionnality requested requires some license and an error was detected when trying to check the current licensing.
)

var resString = map[Status]string{
	Success:        "Success",
	NotInitialized: "NotInitialized",
	AllocFailed:    "AllocFailed",
	InvalidValue:   "InvalidValue",
	ArchMismatch:   "ArchMismatch",
	MappingError:   "MappingError",
	ExecFailed:     "ExecFailed",
	InternalError:  "InternalError",
	Unsupported:    "Unsupported",
	LicenceError:   "LicenceError",
}
