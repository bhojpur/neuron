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
	G "github.com/bhojpur/neuron/pkg/engine"
	"github.com/pkg/errors"
)

var (
	_ Layer = (*Join)(nil)
)

type joinOp int

const (
	composeOp joinOp = iota
	addOp
	elMulOp
)

// Join joins are generalized compositions.
type Join struct {
	Composition
	op joinOp
}

// Add adds the results of two layers/terms.
func Add(a, b Term) *Join {
	return &Join{
		Composition: Composition{
			a: a,
			b: b,
		},
		op: addOp,
	}
}

// HadamardProd performs a elementwise multiplicatoin on the results of two layers/terms.
func HadamardProd(a, b Term) *Join {
	return &Join{
		Composition: Composition{
			a: a,
			b: b,
		},
		op: elMulOp,
	}
}

// Fwd runs the equation forwards.
func (l *Join) Fwd(a G.Input) (output G.Result) {
	if l.op == composeOp {
		return l.Composition.Fwd(a)
	}

	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Forward of a Join %v", l.Name()))
	}

	if l.retVal != nil {
		return l.retVal
	}
	input := a.Node()

	x, err := Apply(l.a, input)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Forward of Join %v - Applying %v to %v failed", l.Name(), l.a, input.Name()))
	}
	xn, ok := x.(*G.Node)
	if !ok {
		return G.Err(errors.Errorf("Expected the result of applying %v to %v to return a *Node. Got %v of %T instead", l.a, input.Name(), x, x))
	}

	y, err := Apply(l.b, input)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Forward of Join %v - Applying %v to %v failed", l.Name(), l.b, input.Name()))
	}
	yn, ok := y.(*G.Node)
	if !ok {
		return G.Err(errors.Errorf("Expected the result of applying %v to %v to return a *Node. Got %v of %T instead", l.a, input.Name(), y, y))
	}

	// perform the op

	switch l.op {
	case addOp:
		return G.LiftResult(G.Add(xn, yn))
	case elMulOp:
		return G.LiftResult(G.HadamardProd(xn, yn))
	}
	panic("Unreachable")
}
