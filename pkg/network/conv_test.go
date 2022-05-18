package network

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

	"github.com/bhojpur/neuron/pkg/engine"
	"github.com/bhojpur/neuron/pkg/tensor"
	"github.com/stretchr/testify/require"
)

func TestConvNet(t *testing.T) {
	c := require.New(t)

	bs := 32
	convKS := tensor.Shape{3, 3}
	mpKS := tensor.Shape{2, 2}

	g := engine.NewGraph()
	x := engine.NewTensor(g, tensor.Float64, 4, engine.WithName("x"), engine.WithShape(bs, 1, 28, 28), engine.WithInit(engine.GlorotU(1)))
	y := engine.NewMatrix(g, tensor.Float64, engine.WithName("y"), engine.WithShape(bs, 10), engine.WithInit(engine.GlorotU(1)))

	nn, err := ComposeSeq(
		x,
		L(ConsConv, WithName("layer 0"), WithSize(bs, 1), WithKernelShape(convKS)),
		L(ConsMaxPool, WithName("layer 0"), WithKernelShape(mpKS)),
		L(ConsDropout, WithName("layer 0"), WithProbability(0.2)),
		L(ConsConv, WithName("layer 1"), WithSize(bs*2, bs), WithKernelShape(convKS)),
		L(ConsMaxPool, WithName("layer 1"), WithKernelShape(mpKS)),
		L(ConsDropout, WithName("layer 1"), WithProbability(0.2)),
		L(ConsConv, WithName("layer 2"), WithSize(bs*4, bs*2), WithKernelShape(convKS)),
		L(ConsMaxPool, WithName("layer 2"), WithKernelShape(mpKS)),
		L(ConsReshape, WithName("layer 2"), ToShape(bs, (bs*4)*3*3)),
		L(ConsDropout, WithName("layer 2"), WithProbability(0.2)),
		L(ConsFC, WithName("layer 3"), WithSize(625), WithActivation(engine.Rectify)),
		L(ConsDropout, WithName("layer 3"), WithProbability(0.55)),
		L(ConsFC, WithName("output"), WithSize(10), WithActivation(SoftMaxFn)),
	)

	c.NoError(err)

	out := nn.Fwd(x)

	err = engine.CheckOne(out)
	c.NoError(err)

	losses := engine.Must(RMS(out, y))
	model := nn.Model()

	_, err = engine.Grad(losses, model...)
	c.NoError(err)

	var costVal engine.Value
	engine.Read(losses, &costVal)

	m := engine.NewTapeMachine(g)

	err = m.RunAll()
	c.NoError(err)

	cost, ok := costVal.Data().(float64)
	c.True(ok)
	c.NotZero(cost)
}
