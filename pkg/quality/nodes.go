package quality

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
	"github.com/bhojpur/neuron/pkg/tensor"
)

// nodes creates new nodes. This will definitely change over time.

// OneHotVector creates a node that represents a one-hot vector.
func OneHotVector(a Class, numClasses uint, dtype tensor.Dtype, opts ...G.NodeConsOpt) *G.Node {
	ohv := ToOneHotVector(a, numClasses, dtype)
	return G.NewConstant(ohv, opts...)
}

// OneHotMatrix creates a node that represents a one-hot matrix.
func OneHotMatrix(a []Class, numClasses uint, dtype tensor.Dtype, opts ...G.NodeConsOpt) *G.Node {
	ohm := ToOneHotMatrix(a, numClasses, dtype)
	return G.NewConstant(ohm, opts...)
}
