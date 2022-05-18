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

// This file provides CGO flags to find CUDA libraries and headers.

//#cgo LDFLAGS:-lcuda
//
////default location:
//#cgo linux LDFLAGS:-L/usr/local/cuda/lib64 -L/usr/local/cuda/lib
//#cgo linux CFLAGS: -I/usr/local/cuda/include
//
////default location if not properly symlinked:
//#cgo linux LDFLAGS:-L/usr/local/cuda-11.0/targets/x86_64-linux/lib
//#cgo linux LDFLAGS:-L/usr/local/cuda-10.2/lib64 -L/usr/local/cuda-10.2/lib
//#cgo linux CFLAGS: -I/usr/local/cuda-11.0/targets/x86_64-linux/include
//#cgo linux CFLAGS: -I/usr/local/cuda-10.2/include/
////Ubuntu 15.04:
//#cgo linux LDFLAGS:-L/usr/lib/x86_64-linux-gnu/
//#cgo linux CFLAGS: -I/usr/include
//
////arch linux:
//#cgo linux LDFLAGS:-L/opt/cuda/lib64 -L/opt/cuda/lib
//#cgo linux CFLAGS: -I/opt/cuda/include
//
////Darwin:
//#cgo darwin LDFLAGS:-L/usr/local/cuda/lib
//#cgo darwin CFLAGS: -I/usr/local/cuda/include/
//
////WINDOWS:
//#cgo windows LDFLAGS:-LC:/cuda/v5.0/lib/x64 -LC:/cuda/v5.5/lib/x64 -LC:/cuda/v6.0/lib/x64 -LC:/cuda/v6.5/lib/x64 -LC:/cuda/v7.0/lib/x64 -LC:/cuda/v8.0/lib/x64 -LC:/cuda/v9.0/x64
//#cgo windows CFLAGS: -IC:/cuda/v5.0/include -IC:/cuda/v5.5/include -IC:/cuda/v6.0/include -IC:/cuda/v6.5/include -IC:/cuda/v7.0/include -IC:/cuda/v8.0/include -IC:/cuda/v9.0/include
import "C"
