package network_test

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
	"fmt"
	"log"

	GG "github.com/bhojpur/neuron/pkg/engine"
	"github.com/bhojpur/neuron/pkg/quality"
	. "github.com/bhojpur/neuron/pkg/quality"
	"github.com/bhojpur/neuron/pkg/tensor"
	"github.com/bhojpur/neuron/pkg/tensor/native"
)

func testFn(emb, sel [][]float64, name string) {
	for i := range sel {
		var s int
		if i == 1 {
			s = 1
		}
		for j := range sel[i] {
			if sel[i][j] != emb[s][j] {
				fmt.Printf("%s error: Expected row %d to be equivalent to row %d of the embedding layer. Expected %v. Got %v instead", name, i, s, emb[s], sel[i])
			}
		}
	}

}

func testFn3(mat [][][]float64, name string) {
	for i := range mat {
		for j := range mat[i] {
			if i == 0 && j == 1 {
				continue
			}
			var eq int
			for k := range mat[i][j] {
				if mat[i][j][k] == mat[0][1][k] { // strictly speaking we should use a better float cmp
					eq++
				}
			}
			if eq >= len(mat[i][j])/2 {
				fmt.Printf("%s Row %d is the same as Row 1.\n%1.1f\n=\n%1.1f\n", name, j, mat[i][j], mat[0][1])
			}
		}

	}
}

func ExampleEmbedding_Defaults() {
	const (
		N        = 13
		dims     = 50
		sentence = 10
	)
	g := GG.NewGraph()
	x := GG.NewVector(g, tensor.Int, GG.WithShape(sentence), GG.WithInit(GG.Zeroes()), GG.WithName("sentence")) // WithInit or WithValue is required.
	w := GG.NewMatrix(g, tensor.Float64, GG.WithShape(N, dims), GG.WithName("embW"), GG.WithInit(GG.GlorotN(1)))
	emb1 := NewEmbedding(WithWeights(w), WithName("emb1"), WithClasses(N), WithBatchSize(sentence))
	emb2 := NewEmbedding(Of(tensor.Float64), WithSize(dims), WithClasses(N), WithBatchSize(sentence))
	sel1 := emb1.Fwd(x)
	sel2 := emb2.Fwd(x)

	classes := x.Value().Data().([]int)
	classes[1] = 1
	vm := GG.NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}
	fmt.Printf("sel1: %v\nsel2: %v\n", sel1.Node().Shape(), sel2.Node().Shape())

	emb1N, err := native.Matrix(emb1.Model()[0].Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	emb2N, err := native.Matrix(emb2.Model()[0].Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	mat1, err := native.Matrix(sel1.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	mat2, err := native.Matrix(sel2.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	// only the 2nd word of the sentence doesn't have word ID 0
	testFn(emb1N.([][]float64), mat1.([][]float64), "mat1")
	testFn(emb2N.([][]float64), mat2.([][]float64), "mat2")

	// Output:
	// sel1: (10, 50)
	// sel2: (10, 50)

}

func ExampleEmbedding_1() {
	const (
		N        = 13
		dims     = 50
		sentence = 10
	)
	g := GG.NewGraph()
	x := GG.NewVector(g, quality.ClassType(), GG.WithShape(sentence), GG.WithInit(GG.Zeroes()), GG.WithName("sentence")) // WithInit or WithValue is required.
	w := GG.NewMatrix(g, tensor.Float64, GG.WithShape(N, dims), GG.WithName("embW"), GG.WithInit(GG.GlorotN(1)))
	emb1 := NewEmbedding(WithWeights(w), WithName("emb1"), WithClasses(N), WithBatchSize(sentence), AsRunner())
	emb2 := NewEmbedding(Of(tensor.Float64), WithSize(dims), WithClasses(N), WithBatchSize(sentence), AsRunner())
	sel1 := emb1.Fwd(x)
	sel2 := emb2.Fwd(x)

	classes := x.Value().Data().([]quality.Class)
	classes[1] = 1

	// Run() is required to be run everytime the VM is run.
	// This is because the embedding layer relies on mechanisms external to the graphs' system.
	if err := emb1.Run(x); err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}
	if err := emb2.Run(x); err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}
	vm := GG.NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}
	fmt.Printf("sel1: %v\nsel2: %v\n", sel1.Node().Shape(), sel2.Node().Shape())

	emb1N, err := native.Matrix(emb1.Model()[0].Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	emb2N, err := native.Matrix(emb2.Model()[0].Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	mat1, err := native.Matrix(sel1.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	mat2, err := native.Matrix(sel2.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	// only the 2nd word of the sentence doesn't have word ID 0
	testFn(emb1N.([][]float64), mat1.([][]float64), "mat1")
	testFn(emb2N.([][]float64), mat2.([][]float64), "mat2")

	// Output:
	// sel1: (10, 50)
	// sel2: (10, 50)

}

func ExampleEmbedding_1_3Tensor() {
	const (
		N         = 13
		dims      = 50
		sentence  = 10
		batchSize = 17
	)
	g := GG.NewGraph()
	x := GG.NewMatrix(g, quality.ClassType(), GG.WithShape(batchSize, sentence), GG.WithInit(GG.Zeroes()), GG.WithName("sentence")) // WithInit or WithValue is required.
	w := GG.NewMatrix(g, tensor.Float64, GG.WithShape(N, dims), GG.WithName("embW"), GG.WithInit(GG.GlorotN(1)))
	emb1 := NewEmbedding(WithWeights(w), WithName("emb1"), WithSize(dims), WithClasses(N), WithBatchSize(batchSize*sentence), AsRunner())
	emb2 := NewEmbedding(Of(tensor.Float64), WithSize(dims), WithClasses(N), WithBatchSize(batchSize*sentence), AsRunner())

	sel1 := emb1.Fwd(x)
	sel2 := emb2.Fwd(x)

	classes, _ := x.Value().Data().([]quality.Class)
	classes[1] = 1

	// Run() is required to be run everytime the VM is run.
	// This is because the embedding layer relies on mechanisms external to the graphs' system.
	if err := emb1.Run(x); err != nil {
		fmt.Println(err)
	}
	if err := emb2.Run(x); err != nil {
		fmt.Println(err)
	}
	vm := GG.NewTapeMachine(g)
	if err := vm.RunAll(); err != nil {
		fmt.Println(err)
		return
	}

	fmt.Printf("sel1: %v\nsel2: %v\n", sel1.Node().Shape(), sel2.Node().Shape())

	mat1, err := native.Tensor3F64(sel1.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	mat2, err := native.Tensor3F64(sel2.Node().Value().(*tensor.Dense))
	if err != nil {
		fmt.Println(err)
	}

	// only the 2nd word of the sentence doesn't have word ID 0
	testFn3(mat1, "mat1")
	testFn3(mat2, "mat2")

	// Output:
	// sel1: (17, 10, 50)
	// sel2: (17, 10, 50)

}
