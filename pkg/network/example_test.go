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

	"github.com/bhojpur/neuron/pkg/engine"
	. "github.com/bhojpur/neuron/pkg/quality"
	"github.com/bhojpur/neuron/pkg/tensor"
)

func softmax(a *engine.Node) (*engine.Node, error) { return engine.SoftMax(a) }

func Example() {
	n := 100
	of := tensor.Float64
	g := engine.NewGraph()
	x := engine.NewTensor(g, of, 4, engine.WithName("X"), engine.WithShape(n, 1, 28, 28), engine.WithInit(engine.GlorotU(1)))
	y := engine.NewMatrix(g, of, engine.WithName("Y"), engine.WithShape(n, 10), engine.WithInit(engine.GlorotU(1)))
	nn, err := ComposeSeq(
		x,
		L(ConsReshape, ToShape(n, 784)),
		L(ConsFC, WithSize(50), WithName("l0"), AsBatched(true), WithActivation(engine.Tanh), WithBias(true)),
		L(ConsDropout, WithProbability(0.5)),
		L(ConsFC, WithSize(150), WithName("l1"), AsBatched(true), WithActivation(engine.Rectify)), // by default WithBias is true
		L(ConsLayerNorm, WithSize(20), WithName("Norm"), WithEps(0.001)),
		L(ConsFC, WithSize(10), WithName("l2"), AsBatched(true), WithActivation(softmax), WithBias(false)),
	)
	if err != nil {
		panic(err)
	}

	out := nn.Fwd(x)
	if err = engine.CheckOne(out); err != nil {
		panic(err)
	}

	cost := engine.Must(RMS(out, y))
	model := nn.Model()
	if _, err = engine.Grad(cost, model...); err != nil {
		panic(err)
	}

	m := engine.NewTapeMachine(g)
	if err := m.RunAll(); err != nil {
		panic(err)
	}

	fmt.Printf("Model: %v\n", model)
	// Output:
	// Model: [l0_W, l0_B, l1_W, l1_B, Norm_W, Norm_B, l2_W]
}
