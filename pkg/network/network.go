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

// ActivationFunction represents an activation function
// Note: This may become an interface once we've worked through all the linter errors
type ActivationFunction func(*G.Node) (*G.Node, error)

// ByNamer is any type that allows a name to be found and returned.
//
// If a name is not found, `nil` is to be returned
type ByNamer interface {
	ByName(name string) Term
}

// Grapher is any type that can return the underlying computational graph
type Grapher interface {
	Graph() *G.ExprGraph
}

// Data represents a layer's data. It is able to reconstruct a Layer and populating it.
type Data interface {
	Make(g *G.ExprGraph, name string) (Layer, error)
}

// Runner is a kind of layer that requires outside-the-graph manipulation.
type Runner interface {
	Run(a G.Input) error

	// Runners should return themselves
	Runnerser
}

// Runnerser is any kind of layer that gets a slice of Runners.
//
// For simplicity, all Runners should implement Runnerser
type Runnerser interface {
	Runners() []Runner
}

// Layer represents a neural network layer.
// λ
type Layer interface {
	// σ - The weights are the "free variables" of a function
	Model() G.Nodes

	// Fwd represents the forward application of inputs
	// x.t
	Fwd(x G.Input) G.Result

	// meta stuff. This stuff is just placholder for more advanced things coming

	Term

	// Serialization stuff

	// Describe returns the protobuf definition of a Layer that conforms to the ONNX standard
	Describe() // some protobuf things TODO
}

type flopser interface {
	FLOPs() int
}

// Redefine redefines a layer with the given construction options. This is useful for re-initializing layers
func Redefine(l Layer, opts ...ConsOpt) (retVal Layer, err error) {
	for _, opt := range opts {
		if l, err := opt(l); err != nil {
			return l, err
		}
	}
	return l, nil
}

// Apply will apply two terms and return the resulting term
// Apply(a, b) has the semantics of a(b).
func Apply(a, b Term) (Term, error) {
	leaveLogScope()
	logf("Apply %v to %v", a.Name(), b.Name())
	enterLogScope()
	defer leaveLogScope()

	var layer Layer
	var retTag bool
	var err error
	switch at := a.(type) {
	case consThunk:
		bInput, ok := b.(G.Input)
		if !ok {
			return nil, errors.Errorf("Applying %v to %v. b is of %T", a, b, b)
		}
		if layer, err = at.LayerCons(bInput, at.Opts...); err != nil {
			return nil, errors.Wrap(err, "Unable to construct layer `a` while calling Apply")
		}
		retTag = true
	case Layer:
		layer = at
	case I:
		layer = nil // identity, return b
	}

	switch bt := b.(type) {
	case G.Input:
		if layer == nil {
			return b, nil
		}

		retVal := layer.Fwd(bt)
		if err = G.CheckOne(retVal); err != nil {
			return nil, errors.Wrap(err, "Apply failed")
		}
		if retTag {
			return tag{layer, retVal.(Term)}, nil
		}
		return retVal.(Term), nil
	default:
		return Compose(b, layer), nil // hmmmmmm this is technically called "stuck". Maybe error?
	}
}
