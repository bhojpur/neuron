package tensor

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
	"sync"

	"gonum.org/v1/gonum/blas"
	"gonum.org/v1/gonum/blas/gonum"
)

var blasdoor sync.Mutex
var whichblas BLAS

// BLAS represents all the possible implementations of BLAS.
// The default is Gonum's Native
type BLAS interface {
	blas.Float32
	blas.Float64
	blas.Complex64
	blas.Complex128
}

// only blastoise.Implementation() and cubone.Implementation() are batchedBLAS -
// they both batch cgo calls (and cubone batches cuda calls)
type batchedBLAS interface {
	WorkAvailable() int
	DoWork()
	BLAS
}

// Use defines which BLAS implementation Bhojpur Neuron should use.
// The default is Gonum's Native. These are the other options:
//		Use(blastoise.Implementation())
//		Use(cubone.Implementation())
//		Use(cgo.Implementation)
// Note the differences in the brackets. The blastoise and cubone ones are functions.
func Use(b BLAS) {
	// close the blast door! close the blast door!
	blasdoor.Lock()
	// open the blast door! open the blast door!
	defer blasdoor.Unlock()
	// those lines were few of the better additions to the Special Edition. There, I said it. The Special Edition is superior. Except Han still shot first in my mind.

	whichblas = b
}

// WhichBLAS returns the BLAS that Bhojpur Neuron uses.
func WhichBLAS() BLAS { return whichblas }

func init() {
	whichblas = gonum.Implementation{}
}
