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

	G "github.com/bhojpur/neuron/pkg/engine"
	. "github.com/bhojpur/neuron/pkg/quality"
	"github.com/bhojpur/neuron/pkg/tensor"
	"github.com/pkg/errors"
)

// myLayer is a layer with additional support for transformation for shapes.
//
// One may of course do this with a ComposeSeq(Reshape, FC), but this is just for demonstration purposes
type myLayer struct {
	// name is in FC
	FC

	// BE CAREFUL WITH EMBEDDINGS

	// size is in FC and in myLayer
	size int
}

// Model, Name, Type, Shape and Describe are all from the embedded FC

func (l *myLayer) Fwd(a G.Input) G.Result {
	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Fwd of myLayer %v", l.FC.Name()))
	}
	x := a.Node()
	xShape := x.Shape()

	switch xShape.Dims() {
	case 0, 1:
		return G.Err(errors.Errorf("Unable to handle x of %v", xShape))
	case 2:
		return l.FC.Fwd(x)
	case 3, 4:
		return G.Err(errors.Errorf("NYI"))

	}
	panic("UNIMPLEMENTED")
}

func ConsMyLayer(x G.Input, opts ...ConsOpt) (retVal Layer, err error) {
	l := new(myLayer)
	for _, opt := range opts {
		var o Layer
		var ok bool
		if o, err = opt(l); err != nil {
			return nil, err
		}
		if l, ok = o.(*myLayer); !ok {
			return nil, errors.Errorf("Construction Option returned non *myLayer. Got %T instead", o)
		}
	}
	if err = l.Init(x.(*G.Node)); err != nil {
		return nil, err
	}
	return l, nil
}

func Example_extension() {
	of := tensor.Float64
	g := G.NewGraph()
	x := G.NewTensor(g, of, 4, G.WithName("X"), G.WithShape(100, 1, 28, 28), G.WithInit(G.GlorotU(1)))
	layer, err := ConsMyLayer(x, WithName("EXT"), WithSize(100))
	if err != nil {
		fmt.Printf("Uh oh. Error happened when constructing *myLayer: %v\n", err)
	}
	l, _ := layer.(*myLayer)
	fmt.Printf("Name:  %q\n", l.Name())
	fmt.Printf("Model: %v\n", l.Model())
	fmt.Printf("BE CAREFUL\n======\nl.size is %v. But the models shapes are correct as follows:\n", l.size)
	for _, n := range l.Model() {
		fmt.Printf("\t%v - %v\n", n.Name(), n.Shape())
	}

	// Output:
	// Name:  "EXT"
	// Model: [EXT_W, EXT_B]
	// BE CAREFUL
	// ======
	// l.size is 0. But the models shapes are correct as follows:
	// 	EXT_W - (1, 100)
	// 	EXT_B - (100, 100)
}
