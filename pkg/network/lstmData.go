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
)

// LSTMData represents a basic LSTM layer
type LSTMData struct {
	inputGateWeight       G.Value
	inputGateHiddenWeight G.Value
	inputBias             G.Value

	forgetGateWeight       G.Value
	forgetGateHiddenWeight G.Value
	forgetBias             G.Value

	outputGateWeight       G.Value
	outputGateHiddenWeight G.Value
	outputBias             G.Value

	cellGateWeight       G.Value
	cellGateHiddenWeight G.Value
	cellBias             G.Value
}

func (l *LSTMData) makeGate(g *G.ExprGraph, name string) lstmGate {
	return makeLSTMGate(
		G.NodeFromAny(g, l.inputGateWeight, G.WithName("wx"+name)),
		G.NodeFromAny(g, l.inputGateHiddenWeight, G.WithName("wh_"+name)),
		G.NodeFromAny(g, l.inputBias, G.WithName("b_"+name)),
	)
}

func (l *LSTMData) Make(g *G.ExprGraph, name string) (Layer, error) {
	var retVal LSTM
	retVal.g = g
	retVal.name = name
	retVal.input = l.makeGate(g, "_input_"+name)
	retVal.forget = l.makeGate(g, "_forget_"+name)
	retVal.output = l.makeGate(g, "_output_"+name)
	retVal.cell = l.makeGate(g, "_cell_"+name)
	return &retVal, nil
}
