package nvrtc_test

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

import (
	"testing"

	"github.com/bhojpur/neuron/pkg/drivers/cu/nvrtc"
)

func TestCompile(t *testing.T) {
	program, err := nvrtc.CreateProgram(`
		extern "C" __global__
		void saxpy(float a, float *x, float *y, float *out, size_t n) {
			size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
			if (tid < n) {
				out[tid] = a * x[tid] + y[tid];
			}
		}
	`, `saxpy.cu`)
	if err != nil {
		t.Fatalf("failed to create program: %v", err)
	}

	err = program.AddNameExpression(`saxpy`)
	if err != nil {
		t.Fatalf("failed to AddNameExpression: %v", err)
	}

	err = program.Compile()
	if err != nil {
		t.Fatalf("failed to Compile: %v", err)
	}

	loweredName, err := program.GetLoweredName(`saxpy`)
	if err != nil {
		t.Fatalf("failed to GetLoweredName: %v", err)
	}
	t.Logf("lowered name: %v", loweredName)

	ptx, err := program.GetPTX()
	if err != nil {
		t.Fatalf("failed to GetPTX: %v", err)
	}
	t.Logf("ptx: %v", ptx)

	programLog, err := program.GetLog()
	if err != nil {
		t.Fatalf("failed to GetLog: %v", err)
	}
	t.Logf("program log: %v", programLog)
}
