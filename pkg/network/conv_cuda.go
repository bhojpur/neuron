//go:build cuda
// +build cuda

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
	"github.com/bhojpur/neuron/pkg/engine/hm"
	"github.com/bhojpur/neuron/pkg/tensor"
)

// Conv represents a convolution layer
type Conv struct{}

// Model will return the engine.Nodes associated with this convolution layer
func (l *Conv) Model() G.Nodes {
	panic("not implemented")
}

// Fwd runs the equation forwards
func (l *Conv) Fwd(x G.Input) G.Result {
	panic("not implemented")
}

// Type will return the hm.Type of the convolution layer
func (l *Conv) Type() hm.Type {
	panic("not implemented")
}

// Shape will return the tensor.Shape of the convolution layer
func (l *Conv) Shape() tensor.Shape {
	panic("not implemented")
}

// Name will return the name of the convolution layer
func (l *Conv) Name() string {
	panic("not implemented")
}

// Describe will describe a convolution layer
func (l *Conv) Describe() {
	panic("not implemented")
}
