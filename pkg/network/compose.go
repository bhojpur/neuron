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
	"fmt"

	G "github.com/bhojpur/neuron/pkg/engine"
	"github.com/bhojpur/neuron/pkg/engine/hm"
	"github.com/bhojpur/neuron/pkg/tensor"
	"github.com/pkg/errors"
)

var (
	_ Layer = (*Composition)(nil)
)

// Composition (∘) represents a composition of functions.
//
// The semantics of ∘(a, b)(x) is b(a(x)).
type Composition struct {
	a, b Term // can be thunk, Layer or *G.Node

	// store returns
	retVal   G.Result
	retType  hm.Type
	retShape tensor.Shape
}

// Compose creates a composition of terms.
func Compose(a, b Term) (retVal *Composition) {
	if _, ok := a.(G.Input); ok {
		a = I{}
	}
	return &Composition{
		a: a,
		b: b,
	}
}

// ComposeSeq creates a composition with the inputs written in left to right order
//
//
// The equivalent in F# is |>. The equivalent in Haskell is (flip (.))
func ComposeSeq(layers ...Term) (retVal *Composition, err error) {
	inputs := len(layers)
	switch inputs {
	case 0:
		return nil, errors.Errorf("Expected more than 1 input")
	case 1:
		// ?????
		return nil, errors.Errorf("Expected more than 1 input")
	}
	l := layers[0]
	for _, next := range layers[1:] {
		l = Compose(l, next)
	}
	return l.(*Composition), nil
}

// Fwd runs the equation forwards
func (l *Composition) Fwd(a G.Input) (output G.Result) {
	if err := G.CheckOne(a); err != nil {
		return G.Err(errors.Wrapf(err, "Forward of a Composition %v", l.Name()))
	}

	if l.retVal != nil {
		return l.retVal
	}
	logf("Compose %v and %v", l.a.Name(), a)
	enterLogScope()
	defer leaveLogScope()
	input := a.Node()

	// apply a to input
	x, err := Apply(l.a, input)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Forward of Composition %v (a)", l.Name()))
	}
	if t, ok := x.(tag); ok {
		l.a, _ = t.a.(Layer)
		x = t.b
	}

	// apply b to the result
	y, err := Apply(l.b, x)
	if err != nil {
		return G.Err(errors.Wrapf(err, "Forward of Composition %v (b)", l.Name()))
	}
	switch yt := y.(type) {
	case tag:
		l.b, _ = yt.a.(Layer)
		retVal, ok := yt.b.(G.Result)
		if !ok {
			return G.Err(errors.Errorf("Error while forwarding Composition where layer is returned. Expected the result of a application to be a Result. Got %v of %T instead", yt.b, yt.b))
		}
		l.retVal = retVal
		return retVal
	case G.Result:
		l.retVal = yt
		return yt
	default:
		return G.Err(errors.Errorf("Error while forwarding Composition. Expected the result of a application to be a Result. Got %v of %T instead", y, y))
	}

}

// Model will return the engine.Nodes associated with this composition
func (l *Composition) Model() (retVal G.Nodes) {
	if a, ok := l.a.(Layer); ok {
		return append(a.Model(), l.b.(Layer).Model()...)
	}
	return l.b.(Layer).Model()
}

// Name will return the name of the composition
func (l *Composition) Name() string {
	aname := "x"
	if l.a != nil {
		aname = l.a.Name()
	}
	return fmt.Sprintf("%v ∘ %v", l.b.Name(), aname)
}

// Describe will describe a composition
func (l *Composition) Describe() { panic("STUB") }

// ByName returns a Term by name
func (l *Composition) ByName(name string) Term {
	if l.a == nil {
		goto next
	}
	if l.a.Name() == name {
		return l.a
	}
next:
	if l.b == nil {
		return nil
	}
	if l.b.Name() == name {
		return l.b
	}
	if bn, ok := l.a.(ByNamer); ok {
		if t := bn.ByName(name); t != nil {
			return t
		}
	}
	if bn, ok := l.b.(ByNamer); ok {
		if t := bn.ByName(name); t != nil {
			return t
		}
	}
	return nil
}

func (l *Composition) Graph() *G.ExprGraph {
	if gp, ok := l.a.(Grapher); ok {
		return gp.Graph()
	}
	if gp, ok := l.b.(Grapher); ok {
		return gp.Graph()
	}
	return nil
}

func (l *Composition) Runners() []Runner {
	var retVal []Runner
	if f, ok := l.a.(Runnerser); ok {
		retVal = append(retVal, f.Runners()...)
	}
	if f, ok := l.b.(Runnerser); ok {
		retVal = append(retVal, f.Runners()...)
	}
	return retVal
}

func (l *Composition) FLOPs() (retVal int) {
	if fa, ok := l.a.(flopser); ok {
		retVal += fa.FLOPs()
	}
	if fb, ok := l.b.(flopser); ok {
		retVal += fb.FLOPs()
	}
	return
}
