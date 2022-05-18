package cu

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

// It provides an idiomatic interface to the CUDA Driver API.

// This file implements CUDA driver context management

//#include <cuda.h>
import "C"
import (
	"fmt"
	"os"
)

const initHtml = "https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__INITIALIZE.html"

func init() {
	// Given that the flags must be 0, the CUDA driver is initialized at the package level
	if err := result(C.cuInit(C.uint(0))); err != nil {
		fmt.Printf("Error in initialization, please refer to %q for details on: %+v\n", initHtml, err)
		os.Exit(1)
	}

}

// Version returns the version of the CUDA driver
func Version() int {
	var v C.int
	if err := result(C.cuDriverGetVersion(&v)); err != nil {
		return -1
	}
	return int(v)
}
