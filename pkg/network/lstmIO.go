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

import G "github.com/bhojpur/neuron/pkg/engine"

// Ensure that lstmInput matches both engine.Input and engine.Result interfaces
var (
	_ G.Input  = &lstmIO{}
	_ G.Result = &lstmIO{}
)

// makeLSTMIO will return a new lstmIO
func makeLSTMIO(x, prevHidden, prevCell *G.Node, err error) (l lstmIO) {
	l.x = x
	l.prevHidden = prevHidden
	l.prevCell = prevCell
	l.err = err

	return l
}

// lstmIO represents an LSTM input/output value
type lstmIO struct {
	x          *G.Node
	prevHidden *G.Node
	prevCell   *G.Node

	err error
}

// Node will return the node associated with the LSTM input
func (l *lstmIO) Node() *G.Node { return nil }

// Nodes will return the nodes associated with the LSTM input
func (l *lstmIO) Nodes() (ns G.Nodes) {
	if l.err != nil {
		return
	}

	return G.Nodes{l.x, l.prevHidden, l.prevCell}
}

// Err will return any error associated with the LSTM input
func (l *lstmIO) Err() error { return l.err }

// Mk makes a new Input, given the xs. This is useful for replacing values in the tuple
//
// CAVEAT: the replacements depends on the length of xs
// 	1: replace x
//	3: replace x, prevCell, prevHidden in this order
//	other: no replacement. l is returned
func (l *lstmIO) Mk(xs ...G.Input) G.Input {
	switch len(xs) {
	case 0:
		return l
	case 1:
		return &lstmIO{x: xs[0].Node(), prevCell: l.prevCell, prevHidden: l.prevHidden}
	case 2:
		return l
	case 3:
		return &lstmIO{x: xs[0].Node(), prevCell: xs[1].Node(), prevHidden: xs[2].Node()}
	default:
		return l
	}
}
